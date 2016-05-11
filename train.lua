require 'load_midi'
require 'common'
require 'paths'
require 'plot_stats'
require 'optim'

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
cmd:option('-L1',0.0008,'L1 decay')
cmd:option('-max_pretrain_epochs', 140, 'number of full passes through the training data while RBM pretrain')

cmd:option('-sparsity_decay_rate',0.9,'decay rate for sparsity')
cmd:option('-sparsity_target',0.08,'sparsity target')
cmd:option('-sparsity_cost',0.0006,'sparsity cost')

cmd:option('-sgd_learning_rate',0.004,'learning rate for SGD')
cmd:option('-sgd_learning_rate_decay',0.97,'learning rate decay')
cmd:option('-sgd_learning_rate_decay_after',40,'in number of epochs, when to start decaying the learning rate')
cmd:option('-max_epochs',80,'number of full passes through the training data')

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

	for i=1, #test do
		test[i] = test[i]:cl()
	end
	for i=1, #train do
		train[i] = train[i]:cl()
	end
elseif opt.cuda then
	require 'cutorch'
	require 'cunn'

	for i=1, #test do
		test[i] = test[i]:cuda()
	end
	for i=1, #train do
		train[i] = train[i]:cuda()
	end
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

-- 1) Run RBM pretrain
criterion = nn.BCECriterion()

function pretrain_feval(t)
	local visible = train[t]

	local pred = rbm:forward(visible)
	local err = criterion:forward(pred, visible)
	rbm:backward(visible)

	dl_dx:mul(-opt.learning_rate)

	L1 = torch.sign(rbm.gradWeight)
	rbm.gradWeight:add(opt.L1, L1)

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

	qval = torch.zeros(opt.n_hidden, 1)

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

	velocity = nn.Module.flatten{weightVelocity, vbiasVelocity, hbiasVelocity}
	x,dl_dx = rbm:getParameters()

	histogramValues = {
	  weight = rbm.weight,
	  vbias = rbm.vbias,
	  hbias = rbm.hbias,

	  weightVelocity = weightVelocity,
	  vbiasVelocity = vbiasVelocity,
	  hbiasVelocity = hbiasVelocity
	}

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
	rnn_inner = nn.LSTM(opt.n_visible, opt.n_recurrent, opt.rho-1)
elseif opt.model == 'gru' then
	rnn_inner = nn.GRU(opt.n_visible, opt.n_recurrent, opt.rho-1)
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

rnn = nn.Sequencer(rnn_inner, opt.rho-1)
mlp = nn.Sequencer(mlp_inner, opt.rho-1)

inputs = {}

if opt.opencl then
	rnn = rnn:cl()
	mlp = mlp:cl()

	for t=1, opt.rho do
		inputs[t] = torch.Tensor(opt.batch_size, roll_height):cl()
	end
elseif opt.cuda then
	rnn = rnn:cuda()
	mlp = mlp:cuda()

	for t=1, opt.rho do
		inputs[t] = torch.Tensor(opt.batch_size, roll_height):cuda()
	end
else
	for t=1, opt.rho do
		inputs[t] = torch.Tensor(opt.batch_size, roll_height)
	end
end

params, gradParam = nn.Container():add(rnn):add(mlp):getParameters()

rnn_learning_rate = opt.sgd_learning_rate

function fine_feval(x_new)
	if params ~= x_new then
		params:copy(x_new)
	end

	load_batch(batch_number, inputs, opt)
	batch_number = batch_number + 1

	gradParam:zero()

	local rnn_inputs = {}
	local mlp_inputs = {}

	for i=1, #inputs-1 do
		rnn_inputs[i]=inputs[i]
	end

	-- 1) prop rnn
	local rnn_outputs = rnn:forward(rnn_inputs)

	-- 2) generate negative phase of RBM
	for i=2, #inputs do
		mlp_inputs[i-1]={inputs[i], rnn_outputs[i-1]}
	end

	mlp:forward(mlp_inputs)

	-- 3) backprop rbm gradients
	local mlp_grads = mlp:backward(mlp_inputs, mlp_inputs)

	-- 4) backprop through time rnn gradients
	local mlp_grads_hid = {}
	for i=1, #mlp_grads do
		mlp_grads_hid[i] = mlp_grads[i][2]
	end

	rnn:backward(rnn_inputs, mlp_grads_hid)

	return _, gradParam
end

-- returns: likelihood,
function evaluate()
	local zeros = torch.zeros(opt.batch_size, opt.n_visible)

	mlp:evaluate()
	rnn:evaluate()

	likelihood = 0
	precision = 0
	recall = 0
	accuracy = 0

	local rnn_output = rnn_inner:forward(zeros)
	for i=1, test_size do
		local pred = mlp_inner:forward{zeros, rnn_output}
		rnn_output = rnn_inner:forward(test[i])

		if pred:ne(pred):sum() > 0 then
			print(sys.COLORS.red .. ' prediction has NaN/s')
		end

		if rnn_output:ne(rnn_output):sum() > 0 then
			print(sys.COLORS.red .. ' hidden rnn has NaN/s')
		end

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

	likelihood = likelihood / test_size
	precision = 100 * precision / test_size
	recall = 100 * recall / test_size
	accuracy = 100 * accuracy / test_size

	fmeasure = (2*precision*recall)/(precision + recall)

	return likelihood, precision, recall, accuracy, fmeasure
end

paths.mkdir('models')

mlp:training()
rnn:training()

for epoch=1, opt.max_epochs do
	print('finetune epoch '..epoch)
	batch_number = 1

	if epoch % 4 == 0 and epoch >= opt.sgd_learning_rate_decay_after then
		rnn_learning_rate = rnn_learning_rate * opt.sgd_learning_rate_decay
		print('decayed learning rate by a factor ' .. opt.sgd_learning_rate_decay .. ' to ' .. rnn_learning_rate)
	end

	local conf = {
		learningRate = rnn_learning_rate,
		alpha = opt.sgd_learning_rate_decay
	}

	for t = 1, train_size do
		optim.rmsprop(fine_feval, params, conf)
		--xlua.progress(t, train_size)

		if params:ne(params):sum() > 0 then
			print(sys.COLORS.red .. ' network params has NaN/s')
		end
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
