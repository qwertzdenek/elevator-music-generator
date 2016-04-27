require 'load_midi'
require 'rbm'
require 'nn'
require 'rnn'
require 'gnuplot'

MIDI=require 'MIDI'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from the NN-RBM music generator model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','train','data directory. Should contain MIDI files.')
-- model params
cmd:option('-rnn_model','models/recurrence-rnn_1.dat','Recurrent module')
cmd:option('-mlp_model','models/recurrence-mlp_1.dat','MLP module with bias and RBM part')

cmd:option('-n_hidden', 50, 'RBM hidden layer size.')
cmd:option('-n_recurrent', 64, 'Recurrent hidden size.')

cmd:option('-length',300,'sample length')
cmd:option('-rho',16,'number of timesteps to unroll for')
cmd:option('-batch_size',10,'number of sequences to train on in parallel')
cmd:option('-o', 'sampled.mid', 'output mid file')

opt = cmd:parse(arg)
torch.seed()

opt.n_visible = roll_height

init_pool(opt)

rnn = torch.load(opt.rnn_model)
mlp = torch.load(opt.mlp_model)

rnn:evaluate()
mlp:evaluate()

piano_roll = torch.Tensor(opt.length, opt.n_visible)
zeros = torch.zeros(opt.n_visible)

-- sample whole piano-roll
rnn_inputs = {}
for i=1, opt.length-1 do
	rnn_inputs[i]=zeros
end
rnn_outputs = rnn:forward(rnn_inputs)

mlp_inputs = {}
for i=2, opt.length do
	mlp_inputs[i-1]={zeros, rnn_outputs[i-1]}
end
mlp_outputs = mlp:forward(mlp_inputs)

for t=1, #mlp_outputs do
    piano_roll[t]:copy(mlp_outputs[t])
end

gnuplot.pngfigure(opt.o..'.png')
gnuplot.imagesc(piano_roll:t())
gnuplot.plotflush()

ticks = 96
division_4 = ticks
division_32 = division_4 / 4

score = {ticks, {}}
score[2][1] = {"patch_change", 0, 0, 0}
score[2][2] = {'set_tempo', 0, 1200000}

counter = 3
for t=1, opt.length do
    for note=1, roll_height do
        if piano_roll[t][note] == 1 then
            -- trace note and erase
            local trace = t
            while trace < opt.length-1 and (piano_roll[trace+1][note] == 1 or piano_roll[trace+2][note] == 1) do
                trace = trace + 1
            end
            local duration = trace - t + 1
            piano_roll[{{t, trace}, note}]:fill(0)
            
            target_note = note + 20
            if target_note >= 0 and target_note <= 127 then
                score[2][counter] = {"note", (t-1)*division_32, duration*division_32, 0, target_note, 90}
                counter = counter + 1
            end
        end
    end
end

midifile = assert(io.open(opt.o,'w'))
midifile:write(MIDI.score2midi(score))
midifile:close()
