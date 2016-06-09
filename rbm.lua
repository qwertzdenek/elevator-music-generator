-- rbm.lua
-- Zdeněk Janeček, 2016 (ycdmdj@gmail.com)
--
-- University of West Bohemia

require 'common'
require 'nn'

local RBM, parent = torch.class('RBM', 'nn.Module')

function RBM:__init(n_visible, n_hidden, batch_size)
	parent.__init(self)
	self.n_visible = n_visible
	self.n_hidden = n_hidden

	self.output = torch.Tensor()
	self.input = torch.Tensor()

	self.weight = torch.Tensor(self.n_hidden, self.n_visible)
	self.vbias = torch.zeros(self.n_visible)
	self.hbias = torch.zeros(self.n_hidden)

	self.gradWeight = torch.zeros(self.n_hidden, self.n_visible)
	self.gradVbias = torch.zeros(self.n_visible)
	self.gradHbias = torch.zeros(self.n_hidden)
	if batch_size ~= nil then
		self.gradVbiasBatch = torch.zeros(batch_size, self.n_visible)
		self.gradHbiasBatch = torch.zeros(batch_size, self.n_hidden)
	end

	self.buffer = torch.Tensor()

	self.posGrad = torch.zeros(self.n_hidden, self.n_visible)
	self.negGrad = torch.zeros(self.n_hidden, self.n_visible)

	self.mu1 = torch.Tensor()
	self.vt = torch.Tensor()

	self.cdSteps = 15

	self:reset()
end

-- propup
function RBM:updateOutputExpected(v_t)
	if v_t:dim() == 1 then
		self.output:resize(self.weight:size(1))
		self.output:copy(self.hbias)
		self.output:addmv(1, self.weight, v_t)
		if self.hbiaslt ~= nil then self.output:add(self.hbiaslt) end
   elseif v_t:dim() == 2 then
		local nframe = v_t:size(1)
		self.output:resize(nframe, self.weight:size(1))
		self.output:mm(v_t, self.weight:t())
		self.output:add(self.hbias:view(1, self.n_hidden)
		  :expand(nframe, self.n_hidden))
		if self.hbiaslt ~= nil then
			self.output:add(self.hbiaslt)
		end
	else
		error('input must be vector or matrix')
	end

	return self.output:sigmoid()
end

-- propdown
function RBM:updateInputExpected(h_t)
	if h_t:dim() == 1 then
		self.input:resize(self.weight:size(2))
		self.input:copy(self.vbias)
		self.input:addmv(1, self.weight:t(), h_t)
		if self.vbiaslt ~= nil then self.input:add(self.vbiaslt) end
	elseif h_t:dim() == 2 then
		local nframe = h_t:size(1)
		self.input:resize(nframe, self.weight:size(2))
		self.input:mm(h_t, self.weight)
		self.input:add(self.vbias:view(1, self.n_visible)
		  :expand(nframe, self.n_visible))
		if self.vbiaslt ~= nil then
			self.input:add(self.vbiaslt)
		end
	else
		error('hidden must be vector or matrix')
	end

	return self.input:sigmoid()
end

-- returns probability of visible activations
function RBM:gibbs(visible)
	local vt = visible
	for t=1, self.cdSteps-1 do
		ht = sample_ber(self:updateOutputExpected(vt))
		vt = sample_ber(self:updateInputExpected(ht))
	end

	ht = sample_ber(self:updateOutputExpected(vt))
	vt = self:updateInputExpected(ht)

	self.vt:resizeAs(vt)
	self.vt:copy(vt)
	sample_ber(self.vt)

	return vt
end

function RBM:updateOutput(input)
	if torch.type(input) == 'table' then
		self.input:resizeAs(input[1])
		self.input:copy(input[1])
		self.vbiaslt = input[2]
		self.hbiaslt = input[3]
	else
		self.input:resizeAs(input)
		self.input:copy(input)
		self.vbiaslt = nil
		self.hbiaslt = nil
	end

	return self:gibbs(self.input)
end

-- we don't need gratOutput in unsupervised greedy training
function RBM:updateGradInput(input)
	if torch.type(input) == 'table' then
		self.input:resizeAs(input[1])
		self.input:copy(input[1])
		self.vbiaslt = input[2]
		self.hbiaslt = input[3]
	else
		self.input:resizeAs(input)
		self.input:copy(input)
		self.vbiaslt = nil
		self.hbiaslt = nil
	end

	local v1 = self.input
	local vt = self.vt

	local mu1 = self:updateOutputExpected(v1)
	self.mu1:resizeAs(mu1)
	self.mu1:copy(mu1)

	local mut = self:updateOutputExpected(vt)

	if v1:dim() == 1 then
		torch.ger(self.posGrad, self.mu1, v1)
		torch.ger(self.negGrad, mut, vt)

		-- update gradients
		torch.csub(self.gradWeight, self.negGrad, self.posGrad)
		torch.csub(self.gradVbias, vt, v1)
		torch.csub(self.gradHbias, mut, self.mu1)

		self.gradInput = {torch.Tensor(), self.gradVbias, self.gradHbias}
	elseif v1:dim() == 2 then
		local nframe = v1:size(1)

		torch.mm(self.posGrad, self.mu1:t(), v1)
		torch.mm(self.negGrad, mut:t(), vt)

		-- update gradients
		torch.csub(self.gradWeight, self.negGrad, self.posGrad)
		self.gradWeight:div(nframe)

		torch.csub(self.gradVbiasBatch, vt, v1)
		torch.mean(self.gradVbias, self.gradVbiasBatch, 1)

		torch.csub(self.gradHbiasBatch, mut, self.mu1)
		torch.mean(self.gradHbias, self.gradHbiasBatch, 1)

		self.gradInput = {torch.Tensor(), self.gradVbiasBatch, self.gradHbiasBatch}
	else
		error('input must be vector or matrix')
	end

	return self.gradInput
end

function RBM:freeEnergy(visible)
	if visible:dim() == 1 then
		self.output:resize(self.n_hidden)
		self.output:copy(self.hbias)
		self.output:addmv(1, self.weight, visible):exp():add(1):log()
		local neg = self.output:sum()
		local pos = torch.dot(visible, self.vbias)
		return -neg-pos
	elseif visible:dim() == 2 then
		local nframe = visible:size(1)
		self.output:resize(nframe, self.n_hidden)
		self.output:mm(visible, self.weight:t())
		self.output:add(self.hbias:view(1, self.n_hidden)
		  :expand(nframe, self.n_hidden))
		self.output:exp():add(1):log()
		local neg = self.output:sum(2)
		local pos = torch.mv(visible, self.vbias)
		return (-neg-pos):sum()
	end
end

function RBM:reset()
	self.weight:normal(0, 0.08)

	self.gradWeight:zero()
	self.gradHbias:zero()
	self.gradVbias:zero()
	self.gradHbiasBatch:zero()
	self.gradVbiasBatch:zero()
end

function RBM:parameters()
   return {self.weight, self.vbias, self.hbias},
		  {self.gradWeight, self.gradVbias, self.gradHbias}
end
