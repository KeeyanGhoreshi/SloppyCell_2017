from pylab import *                                                            # (@\label{whatever is inside these special 'at-parens comments' is visible to LaTeX}\label{code:import_start}@)
from scipy import *
from SloppyCell.ReactionNetworks import *                                      # (@\label{code:import_end}@)

net = IO.from_SBML_file('Ueda.xml', 'net1')                             # (@\label{code:SBML_import}@)
#net.set_var_ic('v1', 'v1_0') # Won't need given initial assignments.           # (@\label{code:SBML_fix}@)

net.set_var_optimizable('L1', False)
net.set_var_optimizable('D1', False)
net.set_var_optimizable('L2', False)
net.set_var_optimizable('D2', False)
net.set_var_optimizable('L3', False)
net.set_var_optimizable('D3', False)
net.set_var_optimizable('L4', False)
net.set_var_optimizable('D4', False)
net.set_var_optimizable('L5', False)
net.set_var_optimizable('D5', False)
net.set_var_optimizable('L6', False)
net.set_var_optimizable('D6', False)
net.set_var_optimizable('L7', False)
net.set_var_optimizable('D7', False)
net.set_var_optimizable('L8', False)
net.set_var_optimizable('D8', False)
net.set_var_optimizable('L9', False)
net.set_var_optimizable('D9', False)
net.set_var_optimizable('L10', False)
net.set_var_optimizable('D10', False)


import data_to_fit_Ueda1                                                               # (@\label{code:import_expt}@)
m = Model([data_to_fit_Ueda1.expt], [net])

params = KeyedList([('a', 1.0),
('A1', 0.45),
('B1', 0.0),
('c1', 0.0),
('r1', 1.02),
('s1', 1.45),
('r', 4.0),
('D0', 0.012),
('A2', 0.45),
('B2', 0.0),
('c2', 0.0),
('r2', 1.02),
('s3', 1.45),
('A3', 0.8),
('B3', 0.6),
('c3', 0.0),
('r3', 0.89),
('s5', 1.63),
('k3', 2.0),
('T3', 1.63),
('k4', 2.0),
('T4', 0.52),
('k2', 2.0),
('T2', 0.72),
('k1', 2.0),
('T1', 1.73),
('v3', 1.63),
('parameter_0000073', 1.63),
('v1', 1.45),
('parameter_0000072', 1.45),
('s4', 0.48),
('s6', 0.47),
('s2', 0.48),
])
print 'Non-optimized parameters:', params

#res = Residuals.PriorInLog('V1_prior', 'V1', 0, log(sqrt(1e4)))                # (@\label{code:prior_start}@)
#m.AddResidual(res)
#res = Residuals.PriorInLog('V2_prior', 'V2', log(4), log(sqrt(4)))
#m.AddResidual(res)                                                             # (@\label{code:prior_end}@)

#print 'Initial cost:', m.cost(params)                                          # (@\label{code:initial_cost}@)
params = Optimization.fmin_lm_log_params(m, params, maxiter=20, disp=False)    # (@\label{code:lm_opt}@)
#print 'Optimized cost:', m.cost(params)
print 'Optimized parameters:', params

# Plot our optimal fit.
figure()                      
Plotting.plot_model_results(m)  # (@\label{code:plot_results}@)

j = m.jacobian_log_params_sens(log(params))                                    # (@\label{code:j_calc}@)
jtj = dot(transpose(j), j)                                                     # (@\label{code:jtj}@)

print 'Beginning ensemble calculation.'
ens, gs, r = Ensembles.ensemble_log_params(m, asarray(params), jtj, steps=7500)# (@\label{code:ens_gen}@)
print 'Finished ensemble calculation.'

show()


pruned_ens = asarray(ens[::25])# (@\label{code:prune}@)

#print len(pruned_ens), len(pruned_ens[0])


index = 0

while index < len(pruned_ens):
    figure()
    hist(log(pruned_ens[:,index]), normed=True) # (@\label{code:hist}@)
    index = index + 1

show()

times = linspace(0, 65, 100)
traj_set = Ensembles.ensemble_trajs(net, times, pruned_ens)                    # (@\label{code:ens_trajs}@)
lower, upper = Ensembles.traj_ensemble_quantiles(traj_set, (0.025, 0.975))

#figure()
#plot(times, lower.get_var_traj('frac_v3'), 'g')
#plot(times, upper.get_var_traj('frac_v3'), 'g')
#plot(times, lower.get_var_traj('frac_v4'), 'b')
#plot(times, upper.get_var_traj('frac_v4'), 'b')

show()
