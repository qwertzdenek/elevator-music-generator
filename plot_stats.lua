require 'gnuplot'

function draw_stats(values, name)
	keys = {'weight', 'vbias', 'hbias', 'weightVelocity', 'vbiasVelocity', 'hbiasVelocity'}
	for _, v in pairs(keys) do
		gnuplot.pngfigure('images/'..v..'-'..name..'.png')
		gnuplot.raw('set yrange [0.1:*]')
		--gnuplot.raw('set ytics ("0" 0.1, "1" 1, "10" 10)')
		gnuplot.raw('set logscale y 2')
		gnuplot.xlabel(string.format('%.6f', torch.abs(values[v]):mean()))
		gnuplot.hist(values[v], 500)
		gnuplot.plotflush()
	end
end

function draw_sc(t, name)
	gnuplot.pngfigure('images/'..name..'.png')
	gnuplot.imagesc(t)
	gnuplot.plotflush()
end

function draw_hist(t, name)
	gnuplot.pngfigure('images/'..name..'.png')
	gnuplot.hist(t)
	gnuplot.plotflush()
end

function draw_plot(t, name)
	gnuplot.pngfigure('images/'..name..'.png')
	gnuplot.plot(t)
	gnuplot.plotflush()
end
