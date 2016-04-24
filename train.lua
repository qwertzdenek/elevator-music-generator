require 'load_midi'
require 'rbm'
require 'common'
require 'image'
require 'paths'
require 'nn'
require 'rnn'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a RNN-RBM music generator model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','train','data directory. Should contain MIDI files.')
-- model params
cmd:option('-n_hidden', 50, 'RBM hidden layer size.')
cmd:option('-n_recurrent', 64, 'Recurrent hidden size.')
cmd:option('-model', 'lstm', 'lstm or gru')
-- optimization
cmd:option('-learning_rate',0.02,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',5,'in number of epochs, when to start decaying the learning rate')
cmd:option('-momentum',0.5,'momentum')

cmd:option('-sparsity_decay_rate',0.9,'decay rate for sparsity')
cmd:option('-sparsity_target',0.07,'sparsity target')
cmd:option('-sparsity_cost',0.055,'sparsity cost')

cmd:option('-rms_learning_rate',0.012,'learning rate for rmsprop')
cmd:option('-rms_decay_rate',0.95,'decay rate for rmsprop')

cmd:option('-rho',16,'number of timesteps to unroll for')
cmd:option('-batch_size',32,'number of sequences to train on in parallel')
cmd:option('-max_epochs',12,'number of full passes through the training data')
cmd:option('-stat_interval',128,'statistics interval')
cmd:option('-init_rbm_from', '', 'initialize RBM from this file')

opt = cmd:parse(arg)
torch.seed()

opt.rms_epsilon = 1e-8
opt.n_visible = roll_height

init_pool(opt)

-- Split dataset
input = torch.zeros(opt.batch_size, roll_height)

test = {}
train = {}
train_size = math.floor(0.7*number_count)
test_size = number_count - train_size

for i=1, test_size do
	load_batch(train_size+i, {input}, opt)
	test[i] = input:clone()
end
for i=1, test_size do
	load_batch(i, {input}, opt)
	train[i] = input:clone()
end

paths.mkdir('models')

function reconstruction_test()
	local err = 0

	for i=1, test_size do
		local v1 = test[i]
		local v2 = rbm:gibbs(v1)
		local diff = torch.csub(v1, v2)
		err = err + diff:abs():mean()
	end

	return err
end

function reconstruction_train()
	local err = 0

	for i=1, test_size do
		local v1 = train[i]
		local v2 = rbm:gibbs(v1)
		local diff = torch.csub(v1, v2)
		err = err + diff:abs():mean()
	end

	return err
end

function free_energy_test()
	local err = 0
	local v1 = torch.Tensor(28*28)

	for i=1, test_size do
		err = err + rbm:freeEnergy(train[i])
	end

	return err
end

function free_energy_train()
	local err = 0
	local v1 = torch.Tensor(28*28)

	for i=1, test_size do
		err = err + rbm:freeEnergy(train[i])
	end

	return err
end

function rmsprop(dx, m, tmp, config)
	-- calculate new (leaky) mean squared values
    m:mul(config.rms_decay_rate)
    m:addcmul(1.0-config.rms_decay_rate, dx, dx)

    -- perform update
    tmp:sqrt(m):add(config.rms_epsilon)
    dx:cdiv(tmp)
end

-- 1) Run RBM pretrain
criterion = nn.BCECriterion()

function pretrain_feval(x_new)
	if x ~= x_new then
		x:copy(x_new)
	end

	local err = 0
	
	load_batch(batch_time, {input}, opt)
	batch_time = batch_time + 1
	
	local pred = rbm:forward(input)
	err = err + criterion:forward(pred, input)
	rbm:backward(input)

	momentum_update(dl_dx, velocity, x, opt)
	sparsity_update(rbm, qval, input, opt)

	return err, dl_dx
end

if string.len(opt.init_rbm_from) > 0 then
	rbmPretrained = torch.load(opt.init_rbm_from)
	rbm = RBM(opt.n_visible, opt.n_hidden, opt.batch_size)
	rbm.weight = rbmPretrained.weight
	rbm.vbias = rbmPretrained.vbias
	rbm.hbias = rbmPretrained.hbias
	
	print('Loaded model from', opt.init_rbm_from)
else
	-- Create RBM
	rbm = RBM(opt.n_visible, opt.n_hidden, opt.batch_size)
	
	-- Training parameters
	weightVelocity = rbm.gradWeight:clone()
	vbiasVelocity = rbm.gradVbias:clone()
	hbiasVelocity = rbm.gradHbias:clone()

	velocity = nn.Module.flatten{weightVelocity, vbiasVelocity, hbiasVelocity}
	x,dl_dx = rbm:getParameters()
	qval = torch.zeros(opt.n_hidden, 1)
	
	batch_time = 1
	err = 0; iter = 0
	for epoch=1, opt.max_epochs do
		print('pretrain epoch '..epoch)
		batch_time = 1
		
		if epoch == math.floor(opt.max_epochs*0.5) then config.momentum = 0.8 end
		if epoch == math.floor(opt.max_epochs*0.72) then config.momentum = 0.9 end
		if epoch == opt.max_epochs then
			torch.save('models/pretrained_rbm_'..epoch..'.dat', rbm)
		end

		for t = 1, train_size do
			iter = iter + 1

			e, dx = pretrain_feval(x)
			err = err + e

			if iter >= opt.stat_interval then
				local test = reconstruction_test(rbm)
				local train = reconstruction_train(rbm)
				local energy_test = free_energy_test(rbm)
				local energy_train = free_energy_train(rbm)

				print(string.format('%s t=%d loss=%.4f test=%.4f train=%.4f ftest=%.4f ftrain=%.4f', os.date("%d/%m %H:%M:%S"), t, err/opt.stat_interval, test, train, energy_test, energy_train))
				
				-- reset counters
				err = 0; iter = 0
			end
		end
	end
end

-- 2) finetune recurrence
if opt.model == 'lstm' then
	rnn = nn.LSTM(opt.n_visible, opt.n_recurrent, opt.rho-1)
elseif opt.model == 'gru' then
	rnn = nn.GRU(opt.n_visible, opt.n_recurrent, opt.rho-1)
else
	error("invalid model type")
end

local bias_inner = nn.ConcatTable()
		:add(nn.LinearNoBias(opt.n_recurrent, opt.n_visible))
		:add(nn.LinearNoBias(opt.n_recurrent, opt.n_hidden))

bias = nn.Recursor(bias_inner, opt.rho-1)

train_size = number_count - 1
rnn_learning_rate = opt.learning_rate

inputs = {}
for t=1, opt.rho do
	inputs[t] = torch.Tensor(opt.batch_size, roll_height)
end

rnnP, rnnG = rnn.recurrentModule:getParameters()
rnnM = torch.Tensor():typeAs(rnnP):resizeAs(rnnG):zero()
rnnTmp = torch.Tensor():typeAs(rnnP):resizeAs(rnnG)

biasP, biasG = bias.recurrentModule:getParameters()
biasM = torch.Tensor():typeAs(biasP):resizeAs(biasG):zero()
biasTmp = torch.Tensor():typeAs(biasP):resizeAs(biasG)

function fine_feval()
    load_batch(batch_time, inputs, opt)
    batch_time = batch_time + 1

    rnn:zeroGradParameters()
    bias:zeroGradParameters()

    -- 1) prop rnn
    rnn_outputs = {}
    for step=1,opt.rho-1 do
      rnn_outputs[step] = rnn:forward(inputs[step])
    end

    -- 2) generate negative phase of RBM
    bias_outputs = {}, {}
    for step=2,opt.rho do
        bias_outputs[step] = bias:forward(rnn_outputs[step-1])
        rbm:forward{inputs[step], bias_outputs[step][1], bias_outputs[step][2]}
    end

    -- 3) estimate rbm gradients
    rbm_grads, mlp_grads = {}, {}
    for step=opt.rho,2,-1 do
		rbm_grads[step] = rbm:updateGradInput{inputs[step], bias_outputs[step][1], bias_outputs[step][2]}
        mlp_grads[step] = bias:backward(rnn_outputs[step-1], {rbm_grads[step][2], rbm_grads[step][3]})
    end

    -- 4) backprop through time gradients
    rnn_grads = {}
    for step=opt.rho-1,1,-1 do
        rnn_grads[step] = rnn:backward(inputs[step], mlp_grads[step+1])
    end

	rnn:forget()
    bias:forget()
	
	rmsprop(rnnG, rnnM, rnnTmp, opt)
	rmsprop(biasG, biasM, biasTmp, opt)

	bias:updateParameters(rnn_learning_rate)
	rnn:updateParameters(rnn_learning_rate)
end

for epoch=1, opt.max_epochs do
	print('finetune epoch '..epoch)
	batch_time = 1

	bias:training()
	rnn:training()
	rnnM:zero()
	biasM:zero()

	if epoch >= opt.learning_rate_decay_after then
		rnn_learning_rate = rnn_learning_rate * opt.learning_rate_decay
		print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. rnn_learning_rate)
	end

    for i = 1, train_size do
        fine_feval()
		xlua.progress(i, train_size)
    end
    
    rnn:evaluate()
	mlpRec:evaluate()
	
	torch.save('models/recurrence-rnn_'..epoch..'.dat', rnn)
	torch.save('models/recurrence-mlp_'..epoch..'.dat', mlpRec)
end
