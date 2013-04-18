set autoscale
set title "No. of processors vs Time taken on a9a dataset"
set xlabel "No. of processors"
set ylabel "Time (in seconds)"
set key right center
set grid
set terminal png
set output "a9a.png"
plot "a9a.graph" using 1:2 with linespoints lt 1 pt 7 title "parallel code timing", \
     417.6 with lines title "sequential code timing = 417.6s"

set autoscale
set title "No. of processors vs Time taken on gisette dataset"
set xlabel "No. of processors"
set ylabel "Time (in seconds)"
set key right center
set grid
set terminal png
set output "gisette.png"
plot "gisette_scale.graph" using 1:2 with linespoints lt 1 pt 7 title "parallel code timing", \
     1138.6 with lines title "sequential code timing = 1138.6s"

set autoscale
set title "No. of processors vs Time taken on w8a dataset"
set xlabel "No. of processors"
set ylabel "Time (in seconds)"
set key right center
set grid
set terminal png
set output "w8a.png"
plot "w8.graph" using 1:2 with linespoints lt 1 pt 7 title "parallel code timing", \
     50.3 with lines title "sequential code timing = 50.3s"

set autoscale
set title "No. of processors vs Time taken on real-sim dataset"
set xlabel "No. of processors"
set ylabel "Time (in seconds)"
set key right center
set grid
set terminal png
set output "real-sim.png"
plot "real-sim.graph" using 1:2 with linespoints lt 1 pt 7 title "parallel code timing", \
      5768.7 with lines title "sequential code timing = 5768.7s"

