require 'load_midi'
require 'common'
require 'paths'
require 'plot_stats'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a RNN-RBM music generator model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','train','data directory. Should contain MIDI files.')
cmd:option('-prefix', '', 'prefix of this run')
cmd:option('-v', false, 'verbose mode')

-- model params
cmd:text('Model parameters')
cmd:option('-n_hidden', 150, 'RBM hidden layer size.')
cmd:option('-n_recurrent', 100, 'Recurrent hidden size.')
cmd:option('-model', 'lstm', 'lstm or gru')
cmd:option('-init_rbm_from', '', 'initialize RBM from this file')

cmd:text('Optimalization parameters')
-- optimization
cmd:option('-learning_rate',0.018,'learning rate')
cmd:option('-momentum',0.5,'momentum')
cmd:option('-max_pretrain_epochs', 50, 'number of full passes through the training data while RBM pretrain')

cmd:option('-sparsity_decay_rate',0.9,'decay rate for sparsity')
cmd:option('-sparsity_target',0.08,'sparsity target')
cmd:option('-sparsity_cost',0.0012,'sparsity cost')

cmd:option('-sgd_learning_rate',0.004,'learning rate for SGD')
--cmd:option('-rmsprop_decay_rate',0.95,'decay rate for RMSProp')
cmd:option('-sgd_learning_rate_decay',0.97,'learning rate decay')
cmd:option('-sgd_learning_rate_decay_after',40,'in number of epochs, when to start decaying the learning rate')
cmd:option('-max_epochs',140,'number of full passes through the training data')

cmd:option('-rho',32,'number of timesteps to unroll for')
cmd:option('-batch_size',100,'number of sequences to train on in parallel')
cmd:option('-stat_interval',256,'statistics interval')
cmd:option('-opencl', false,'use OpenCL backend')
cmd:option('-cuda', false,'use CUDA backend')

opt = cmd:parse(arg)
torch.seed()

--~ opt.rmsprop_epsilon = 1e-8
opt.n_visible = roll_height

init_pool(opt)

-- Split dataset
input = torch.DoubleTensor(opt.batch_size, roll_height)

test = {}
train = {}

-- 512 is enough
number_count = math.min(512, number_count)
train_size = math.floor(0.7*number_count)
test_size = number_count - train_size

print('Train sample size', train_size)
print('Test sample size', test_size)

for i=1, test_size do
	load_batch(train_size+i, {input}, opt)
	test[i] = input:clone()
end
for i=1, train_size do
	load_batch(i, {input}, opt)
	train[i] = input:clone()
end

if opt.opencl then
	require 'cltorch'
	require 'clnn'

	cltorch.setTrace(1)

	test = test:cl()
	train = train:cl()
elseif opt.cuda then
	require 'cutorch'
	require 'cunn'

	test = test:cuda()
	train = train:cuda()
else
	require 'nn'
end

require 'rbm'
require 'rnn'

function reconstruction_test()
	local err = 0

	for i=1, test_size do
		local v1 = test[i]
		local v2 = rbm:gibbs(v1)
		local diff = torch.csub(v1, v2)
		err = err + diff:abs():mean()
	end

	return (err/test_size)*100
end

function reconstruction_train()
	local err = 0

	for i=1, train_size do
		local v1 = train[i]
		local v2 = rbm:gibbs(v1)
		local diff = torch.csub(v1, v2)
		err = err + diff:abs():mean()
	end

	return (err/train_size)*100
end

function free_energy_test()
	local err = 0
	local v1 = torch.Tensor(28*28)

	for i=1, test_size do
		err = err + rbm:freeEnergy(test[i])
	end

	return err/test_size
end

function free_energy_train()
	local err = 0
	local v1 = torch.Tensor(28*28)

	for i=1, train_size do
		err = err + rbm:freeEnergy(train[i])
	end

	return err/train_size
end

--~ function rmsprop(dx, m, tmp, config)
	--~ -- calculate new (leaky) mean squared values
	--~ m:mul(config.rmsprop_decay_rate)
	--~ m:addcmul(1.0-config.rmsprop_decay_rate, dx, dx)

	--~ -- perform update
	--~ tmp:sqrt(m):add(config.rmsprop_epsilon)
	--~ dx:cdiv(tmp)
--~ end

-- 1) Run RBM pretrain
criterion = nn.BCECriterion()

function pretrain_feval(t)
	local visible = train[t]

	local pred = rbm:forward(visible)
	local err = criterion:forward(pred, visible)
	rbm:backward(visible)

	momentum_update(dl_dx, velocity, x, opt)
	sparsity_update(rbm, qval, visible, opt)

	return err
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

	histogramValues = {
	  weight = rbm.weight,
	  vbias = rbm.vbias,
	  hbias = rbm.hbias,

	  weightVelocity = weightVelocity,
	  vbiasVelocity = vbiasVelocity,
	  hbiasVelocity = hbiasVelocity
	}

	if opt.opencl then
		criterion = criterion:cl()
		rbm = rbm:cl()
		weightVelocity = weightVelocity:cl()
		vbiasVelocity = vbiasVelocity:cl()
		hbiasVelocity = hbiasVelocity:cl()
		qval = qval:cl()
	end
	if opt.cuda then
		criterion = criterion:cuda()
		rbm = rbm:cuda()
		weightVelocity = weightVelocity:cuda()
		vbiasVelocity = vbiasVelocity:cuda()
		hbiasVelocity = hbiasVelocity:cuda()
		qval = qval:cuda()
	end

	err = 0; iter = 0
	for epoch=1, opt.max_pretrain_epochs do
		print('pretrain epoch '..epoch)

		velocity:zero()

		if epoch == math.floor(opt.max_pretrain_epochs*0.5) then
			torch.save('models/'..opt.prefix..'pretrained_rbm_'..epoch..'.dat', rbm)
			config.momentum = 0.8
		end
		if epoch == math.floor(opt.max_pretrain_epochs*0.72) then
			config.momentum = 0.9
		end
		if epoch == opt.max_pretrain_epochs then
			torch.save('models/'..opt.prefix..'pretrained_rbm_'..epoch..'.dat', rbm)
		end

		for t = 1, train_size do
			iter = iter + 1

			err = err + pretrain_feval(t)

			if iter >= opt.stat_interval then
				local test = reconstruction_test(rbm)
				local train = reconstruction_train(rbm)
				local energy_test = free_energy_test(rbm)
				local energy_train = free_energy_train(rbm)

				print(string.format('%s t=%d loss=%.4f test=%.4f%% train=%.4f%% ftest=%.4f ftrain=%.4f', os.date("%d/%m %H:%M:%S"), t, err/opt.stat_interval, test, train, energy_test, energy_train))

				-- reset counters
				err = 0; iter = 0

				if opt.v then
					draw_hist(rbm.mu1:mean(1), 'mean_hidden-'..epoch..'-'..t, 'pravděpodobnost')
				end
			end
		end

		if opt.v then
			draw_stats(histogramValues, 'hist_'..epoch)

			gnuplot.pngfigure('images/weight-map_'..epoch..'.png')
			gnuplot.ylabel('skryté')
			gnuplot.xlabel('viditelné')
			gnuplot.imagesc(rbm.weight)
			gnuplot.plotflush()
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

-- {input(t), output(t-1)} -> outputV(t)
mlp_inner = nn.Sequential()
	:add(nn.ParallelTable()
		:add(nn.Identity())
		:add(
			nn.ConcatTable()
				:add(nn.LinearNoBias(opt.n_recurrent, opt.n_visible))
				:add(nn.LinearNoBias(opt.n_recurrent, opt.n_hidden))
		)
	)
	:add(nn.FlattenTable())
	:add(rbm)

mlp = nn.Recursor(mlp_inner, opt.rho-1)

rnn_learning_rate = opt.sgd_learning_rate

inputs = {}
for t=1, opt.rho do
	inputs[t] = torch.Tensor(opt.batch_size, roll_height)
end

-- RMSProp parameters, in case it is solved in upstream
--~ rnnP, rnnG = rnn:getParameters()
--~ rnnM = torch.Tensor():typeAs(rnnP):resizeAs(rnnG):zero()
--~ rnnTmp = torch.Tensor():typeAs(rnnP):resizeAs(rnnG)

--~ mlpP, mlpG = mlp:getParameters()
--~ mlpM = torch.Tensor():typeAs(mlpP):resizeAs(mlpG):zero()
--~ mlpTmp = torch.Tensor():typeAs(mlpP):resizeAs(mlpG)

if opt.opencl then
	rnn = rnn:cl()
	mlp = mlp:cl()

	inputs = inputs:cl()
end
if opt.cuda then
	rnn = rnn:cuda()
	mlp = mlp:cuda()

	inputs = inputs:cuda()
end

function fine_feval(t)
	load_batch(t, inputs, opt)

	rnn:zeroGradParameters()
	mlp:zeroGradParameters()

	rnn:forget()
	mlp:forget()

	rnn_outputs, mlp_outputs, mlp_grads  = {}, {}, {}
	-- 1) prop rnn
	for step=1,opt.rho-1 do
		rnn_outputs[step] = rnn:forward(inputs[step])
	end

	-- 2) generate negative phase of RBM
	for step=2,opt.rho do
		mlp:forward{inputs[step], rnn_outputs[step-1]}
	end

	-- 3) backprop rbm gradients
	for step=opt.rho,2,-1 do
		mlp_grads[step] = mlp:backward{inputs[step], rnn_outputs[step-1]}
	end

	-- 4) backprop through time gradients
	for step=opt.rho-1,1,-1 do
		rnn:backward(inputs[step], mlp_grads[step+1][2])
	end

	--rmsprop(rnnG, rnnM, rnnTmp, opt)
	--rmsprop(mlpG, mlpM, mlpTmp, opt)

	mlp:updateParameters(rnn_learning_rate)
	rnn:updateParameters(rnn_learning_rate)
end

-- returns: likelihood,
function evaluate()
	mlp:evaluate()
	rnn:evaluate()

	likelihood = 0
	precision = 0
	recall = 0
	accuracy = 0

	local rnn_output = rnn:forward(torch.zeros(opt.batch_size, opt.n_visible))
	for i=1, test_size do
		local pred = mlp:forward{test[i], rnn_output}
		rnn_output = rnn:forward(test[i])

		likelihood = likelihood + criterion:forward(pred, test[i])

		local TP = torch.cmul(pred, test[i]):sum()
		local FP = torch.cmul(pred:byte(), torch.eq(test[i], 0)):sum()
		local FN = torch.cmul(torch.eq(pred, 0), test[i]:byte()):sum()

		precision = precision + TP / (TP + FP)
		recall = recall + TP / (TP + FN)
		accuracy = accuracy + TP / (TP + FP + FN)
	end

	mlp:training()
	rnn:training()

	fmeasure = (2*precision*recall)/(precision + recall)

	return likelihood, precision, recall, accuracy, fmeasure
end

paths.mkdir('models')

mlp:training()
rnn:training()

for epoch=1, opt.max_epochs do
	print('finetune epoch '..epoch)

	if epoch % 4 == 0 and epoch >= opt.sgd_learning_rate_decay_after then
		rnn_learning_rate = rnn_learning_rate * opt.sgd_learning_rate_decay
		print('decayed learning rate by a factor ' .. opt.sgd_learning_rate_decay .. ' to ' .. rnn_learning_rate)
	end

	for t = 1, train_size do
		fine_feval(t)
		--xlua.progress(t, train_size)
	end

	likelihood, precision, recall, accuracy, fmeasure = evaluate()

	print(string.format('  log-likelihood=%.4f', -likelihood))
	print(string.format('  Precision=%.4f', precision))
	print(string.format('  Recall=%.4f', recall))
	print(string.format('  Accuracy=%.4f', accuracy))
	print(string.format('  F-measure=%.4f', fmeasure))

	if epoch % 10 == 0 then
		rnn:forget()
		mlp:forget()

		torch.save('models/'..opt.prefix..'recurrence-rnn_'..epoch..'.dat', rnn)
		torch.save('models/'..opt.prefix..'recurrence-mlp_'..epoch..'.dat', mlp)
	end
end

rnn:forget()
mlp:forget()
