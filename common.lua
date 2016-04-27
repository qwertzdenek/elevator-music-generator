require 'nn'

target = torch.Tensor(1)

function momentum_update(derivate, velocity, target, config)
	derivate:mul(-config.learning_rate)
	velocity:mul(config.momentum):add(derivate)
	target:add(velocity)
end

function sample_ber(x)
	x:csub(torch.rand(x:size())):sign():clamp(0, 1)
	return x
end

function sparsity_update(rbm, qold, input, config)
	-- get moving average of last value and current
	qcurrent = rbm.mu1:mean(1)[1]
	qcurrent:mul(1-config.sparsity_decay_rate)
    qcurrent:add(config.sparsity_decay_rate, qold)
    qold:copy(qcurrent)

    target:resizeAs(qcurrent)
    target:fill(config.sparsity_target)
    diffP = qcurrent:csub(target)
    dP_dW = torch.ger(diffP, input:mean(1)[1])
    
    rbm.weight:csub(dP_dW:mul(config.sparsity_cost))
    rbm.hbias:csub(diffP:mul(config.sparsity_cost))
end
