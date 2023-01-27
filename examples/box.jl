#= Enzyme for adjoint tutorial: Stommel three-box ocean model

The goal of this tutorial is to teach about a specific usage of Enzyme's automatic 
differentiation capabilities, and will be centered around the Stommel ocean model. This 
is a nice example to see how powerful Enzyme is, and the ability of it to take a 
derivative of a complicated function (namely one that has many parts and parameters).
This tutorial will focus first on the computations and getting Enzyme running, 
for those interested a mathematical explanation of the model and what an adjoint 
variable is will be provided at the end. 

# Brief model overview 

The Stommel box model can be viewed as a watered down full ocean model. In our example, we have three
boxes (Box One, Box Two, and Box Three) and we model the transport of fluid between 
them. The full equations of our system are given by:

```math
\begin{aligned}
   U &= u_0 \left\{ \rho_2 - \left[ \rho_1 + (1 - \delta) \rho_3 \right] \right\} \\
   \rho_i &= -\alpha T_i + \beta S_i, \; \; \; \; i = 1, 2, 3
\end{aligned}
```

for the **transport** U and **densities** ``\rho``, and then the time derivatives

```math
\begin{aligned}
   \dot{T_1} &= U(T_3 - T_1)/V_1 + \gamma (T_1^* - T_1 ) & \dot{S_1} &= U(S_3 - S_1)/V_1 + FW_1/V_1 \\
   \dot{T_2} &= U(T_1 - T_2)/V_2 + \gamma (T_2^* - T_2 ) & \dot{S_2} &= U(S_1 - S_2)/V_2 + FW_2/V_2 \\
   \dot{T_3} &= U(T_2 - T_3)/V_3 & \dot{S_3} &= U(S_2 - S_3)/V_3 
\end{aligned}
```

for positive transport, ``U > 0``, and 

```math
\begin{aligned}
   \dot{T_1} &= U(T_2 - T_1)/V_1 + \gamma (T_1^* - T_1) & \dot{S_1} &= U(S_2 - S_1)/V_1 + FW_1/V_1 \\
   \dot{T_2} &= U(T_3 - T_2)/V_2 + \gamma (T_2^* - T_2 ) & \dot{S_2} &= U(S_3 - S_2)/V_2 + FW_2/V_2 \\
   \dot{T_3} &= U(T_1 - T_3)/V_3 & \dot{S_3} &= U(S_1 - S_3)/V_3 
\end{aligned}
```

for ``U \leq 0``.
The only force driving our system is a density gradient generated via temperature
and salinity differences between the boxes. This makes it a really easy 
model to play around with! With this in mind, the model is run 
forward with the steps:

1) Compute densities
2) Compute transport 
3) Compute time derivatives of the box temperatures and salinities
4) Update the state vector

We'll start by going through the model setup step by step, then providing a few test 
cases with Enzyme.  

# Model setup 

## Model dependencies

# Let's first add the necessary packages to run everything 
=#

import Pkg; Pkg.activate(".")
using Enzyme
using Unitful

# ## Initialize constants

# The system equations have quite a few constants that appear, here we initialize them 
# for later use 

const yr = u"yr"
const d = u"d"
const cm = u"cm"
const g =u"g"
const kg = u"kg"
const °C = u"°C"
const K = u"K"
const m = u"m"
const km = u"km"
const m³ = m^3
const s = u"s"

ENV["UNITFUL_FANCY_EXPONENTS"] = true

const blength = ([5000.0; 1000.0; 5000.0])km   ## north-south size of boxes, centimeters

const bdepth = ([1.0; 5.0; 4.0])km   ## depth of boxes, centimeters
  
const δ = bdepth[1]/(bdepth[1] + bdepth[3])  ## constant ratio of two depths

const bwidth = 4000.0km  ## box width, centimeters

## box areas
const barea = [blength[1]*bwidth;
         blength[2]*bwidth;
         blength[3]*bwidth]

## box volumes
const bvol = [barea[1]*bdepth[1];
        barea[2]*bdepth[2];
        barea[3]*bdepth[3]]

## parameters that are used to ensure units are in CGS (cent-gram-sec)

const hundred = 100
const thousand = 1000
module UnitfulOcean; using Unitful; @unit Sverdrup "Sv" Sverdrup (10^6)u"m^3/s" false; end
Unitful.register(UnitfulOcean);
Sv = u"Sverdrup"

## parameters that appear in box model equations
uc = (16.0/0.004)
u0 = (Sv/(g/cm^3))uc  # units error in Tziperman/Ioannou 2002
α = 1668e-7(g/cm^3/K)
β = 0.7811e-3*(g/cm^3)/(g/kg)
γ = 1/(300d)

## robert filter coefficient for the smoother part of the timestep 
const robert_filter_coeff = 0.25 

S₀ = 35.0g/kg

## freshwater forcing, convert to useful units as well
FW = (Sv*g/kg).([100.0cm/yr * S₀ * barea[1]; -100.0cm/yr * S₀ * barea[1]])

## restoring atmospheric temperatures
T★ = ([22.0, 0.0])°C # J + \bigstar + TAB
S★ = ([36.0; 34.0])g/kg 

## handle equation of state and affine temperature units
state_ref = vcat(fill(0.0°C,3),fill(0.0g/kg,3))

# ## Define model functions 

# Here we define functions that will calculate quantities used in the forward steps.

"""
     function to compute transport
       Input: rho - the density vector
       Output: U - transport value 
"""
function U_func(ρ)

    U = u0*(ρ[2] - (δ * ρ[1] + (1 - δ)*ρ[3])) 
    return U |> Sv

end

"""
 function to compute density
       Input: state = [T1; T2; T3; S1; S2; S3]
       Output: rho 
"""
function rho_func(state)

    ρ = fill(0.0kg/m³,3)

    # deal with kelvin, not degrees C
    Δstate = state .- state_ref
    ρ[1] = -α * Δstate[1] + β * Δstate[4]
    ρ[2] = -α * Δstate[2] + β * Δstate[5]
    ρ[3] = -α * Δstate[3] + β * Δstate[6]

    return ρ

end

"""
     lastly our timestep function
      Input: fld_now = [T1(t), T2(t), ..., S3(t)]
          fld_old = [T1(t-dt), ..., S3(t-dt)]
          u = transport(t)
          dt = time step
      Output: fld_new = [T1(t+dt), ..., S3(t+dt)]
"""
function timestep_func(fld_now, fld_old, u, dt)

    
    temp = zeros(6).*unit.(fld_now)
    fld_new = zeros(6).*unit.(fld_now)

    ## first computing the time derivatives of the various temperatures and salinities
    if u > 0Sv

        temp[1] = u * (fld_now[3] - fld_now[1]) / bvol[1] + γ * (T★[1] - fld_now[1]) 
        temp[2] = u * (fld_now[1] - fld_now[2]) / bvol[2] + γ * (T★[2] - fld_now[2])
        temp[3] = u * (fld_now[2] - fld_now[3]) / bvol[3] 

        temp[4] = u * (fld_now[6] - fld_now[4]) / bvol[1] + FW[1] / bvol[1]
        temp[5] = u * (fld_now[4] - fld_now[5]) / bvol[2] + FW[2] / bvol[2]
        temp[6] = u * (fld_now[5] - fld_now[6]) / bvol[3]

    elseif u <= 0Sv

        temp[1] = u * (fld_now[2] - fld_now[1]) / bvol[1] + γ * (T★[1] - fld_now[1]) 
        temp[2] = u * (fld_now[3] - fld_now[2]) / bvol[2] + γ * (T★[2] - fld_now[2])

        temp[3] = u * (fld_now[1] - fld_now[3]) / bvol[3] 

        temp[4] = u * (fld_now[5] - fld_now[4]) / bvol[1] + FW[1] / bvol[1]
        temp[5] = u * (fld_now[6] - fld_now[5]) / bvol[2] + FW[2] / bvol[2]
        temp[6] = u * (fld_now[4] - fld_now[6]) / bvol[3]

    end

    ## update fldnew using a version of Euler's method  

    for j = 1:6
        fld_new[j] = fld_old[j] + 2dt * temp[j] 
    end 
    
    return uconvert.(unit.(fld_old),fld_new) # affine units 
end 

"""
    Define forward functions

    Finally, we create two functions, the first of which computes and stores all the 
    states of the system, and the second which has been written specifically to be 
    passed to Enzyme. 

    Let's start with the standard forward function. This is just going to be used
    to store the states at every timestep:
"""
function forward_func(fld_old, fld_now, dt, M)

    state_now = copy(fld_now)
    state_old = copy(fld_old)
    state_new = zeros(6).*unit.(state_now)

    states_unsmooth = [state_old]                      
    states_smooth = [state_old]

    for t = 1:M
        ρ_now = rho_func(state_now)
        u_now = U_func(ρ_now)
        state_new = timestep_func(state_now, state_old, u_now, dt)

        ## Robert filter smoother (needed for stability)
        # re-arrange to accommodate affine quantities
        for j = 1:6
            state_now[j] = state_now[j] + robert_filter_coeff * ((state_new[j] - state_now[j]) + (state_old[j]- state_now[j]))
        end 

        push!(states_smooth, copy(state_now))
        push!(states_unsmooth, copy(state_new))

        ## cycle the "now, new, old" states 

        state_old = state_now 
        state_now = state_new 
    end

    return states_smooth, states_unsmooth
end

"""
     Next, we have the Enzyme-designed forward function. This is what we'll actually
     be passing to Enzyme to differentiate: 
"""
function forward_func_4_AD(in_now, in_old, out_old, out_now)

    ρ_now = rho_func(in_now)                             ## compute density
    u_now = U_func(ρ_now)                                ## compute transport 
    in_new = timestep_func(in_now, in_old, u_now, 10d)  ## compute new state values

    ## Robert filter smoother 
    in_now[1] = in_now[1] + robert_filter_coeff * (in_new[1] - in_now[1] + in_old[1] - in_now[1])
    in_now[2] = in_now[2] + robert_filter_coeff * (in_new[2] -  in_now[2] + in_old[2] -in_now[2])
    in_now[3] = in_now[3] + robert_filter_coeff * (in_new[3] - in_now[3] + in_old[3] -in_now[3])
    in_now[4] = in_now[4] + robert_filter_coeff * (in_new[4] - in_now[4] + in_old[4] - in_now[4])
    in_now[5] = in_now[5] + robert_filter_coeff * (in_new[5] - in_now[5] + in_old[5] - in_now[5])
    in_now[6] = in_now[6] + robert_filter_coeff * (in_new[6] - in_now[6] + in_old[6] - in_now[6])

    out_old[:] = in_now 
    out_now[:] = in_new
    return nothing

end

#= Two key differences:
1) `forward_func_4_AD` now returns nothing, but is rather a function of both its input
   and output.
2) All operations are now inlined, meaning we compute the entries of the input vector
   individually.
Currently, Enzyme does not have compatibility with matrix/vector operations so inlining
is necessary to run Enzyme on this function. 

# Example 1: Simply using Enzyme

For the first example let's just compute the gradient of our forward function and 
examine the output. We'll just run the model for one step, and take a `dt` of ten 
days. The initial conditions of the system are given as `Tbar` and `Sbar`. 
=#

T̄ = ([20.0; 1.0; 1.0])°C
S̄ = ([35.5; 34.5; 34.5])g/kg
   
## Running the model one step forward
Δt = 10d
state_bar = copy([T̄; S̄])
states_smooth, states_unsmooth = forward_func(state_bar, state_bar, Δt, 1)
    
## Run Enzyme one time on `forward_func_4_AD``

# can't invert °C because it is affine
d_units = vcat(fill(K^-1,3),fill(kg/g,3))
din_now = zeros(6).*d_units
din_old = zeros(6).*d_units
out_now = zeros(6).*unit.(state_bar)
dout_now = ones(6).*d_units
out_old = zeros(6).*unit.(state_bar)
dout_old = ones(6).*d_units

autodiff(forward_func_4_AD, Duplicated([T̄; S̄], din_now), Duplicated([T̄; S̄], din_old), 
                    Duplicated(out_now, dout_now), Duplicated(out_old, dout_old));

# In order to run Enzyme on `forward_func_4_AD`, we've needed to provide quite a few 
# placeholders, and wrap everything in [`Duplicated`](@ref) as all components of our function 
# are vectors, not scalars. Let's go through and see what Enzyme did with all 
# of those placeholders. 

# First we can look at what happened to the zero vectors out_now and out_old:

@show out_now, out_old

# Comparing to the results of forward func:

@show states_smooth[2], states_unsmooth[2]

# we see that Enzyme has computed and stored exactly the output of the 
# forward step. Next, let's look at `din_now`: 

@show din_now 

# Just a few numbers, but this is what makes AD so nice: Enzyme has exactly computed
# the derivative of all outputs with respect to the input in_now, evaluated at
# in_now, and acted with this gradient on what we gave as dout_now (in our case, 
# all ones). In math language, this is just
# ```math 
# \text{din now} = (\frac{\partial \text{out now}(\text{in now})}{\partial \text{in now}} + \frac{\partial \text{out old}(\text{in now})}{\partial \text{in now}}) \text{dout now} 
# ```

# We note here that had we given dout_now and dout_now as something else, our results 
# will change. Let's multiply them by two and see what happens. 

din_now_new = zeros(6)
din_old_new = zeros(6)
out_now = zeros(6); dout_now = fill(2,6) 
out_old = zeros(6); dout_old = fill(2,6)
autodiff(forward_func_4_AD, Duplicated([T̄; S̄], din_now_new), Duplicated([T̄; S̄], din_old_new), 
                    Duplicated(out_now, dout_now), Duplicated(out_old, dout_old));

# Now checking din_now_new and din_old_new we see

@show din_now_new

# What happened? Enzyme is actually taking the computed gradient and acting on what we
# give as input to dout_now and dout_old. Checking this, we see

@show 2*din_now

# and they match the new results.

# # Example 2: Full sensitivity calculations

# Now we want to use Enzyme for a bit more than just a single derivative. Let's 
# say we'd like to understand how sensitive the final temperature of Box One is to the initial 
# salinity of Box Two. That is, given the function 
#
# ```math
# J = (1,0,0,0,0,0)^T \cdot \mathbf{x}(t_f)
# ```
# we want Enzyme to calculate the derivative 
#
# ```math
# \frac{\partial J}{\partial \mathbf{x}(0)}
# ```
#
# where ``x(t)`` is the state of the model at time t. If we think about ``x(t_f)`` as solely depending on the 
# initial condition, then this derivative is really 
#
# ```math
# \frac{\partial J}{\partial \mathbf{x}(0)} = \frac{\partial}{\partial \mathbf{x}(0)} \left( (1,0,0,0,0,0)^T \cdot L(\ldots(L(\mathbf{x}(0)))) \right) 
# ```
#
# with ``L(x(t)) = x(t + dt)``, i.e. one forward step. One could expand this derivative with the chain rule (and it would be very complicated), but really this 
# is where Enzyme comes in. Each run of autodiff on our forward function is one piece of this big chain rule done for us! We also note that the chain rule
# goes from the outside in, so we start with the derivative of the forward function at the final state, and work backwards until the initial state.
# To get Enzyme to do this, we complete the following steps:
# 1) Run the forward model and store outputs (in a real ocean model this wouldn't be 
#       feasible and we'd need to use checkpointing)
# 2) Compute the initial derivative from the final state 
# 3) Use Enzyme to work backwards until we reach the desired derivative. 
#
# For simplicity we define a function that takes completes our AD steps

function ad_calc(in_now, in_old, M)

dout_old = [1.0;0.0;0.0;0.0;0.0;0.0]
dout_now = [0.0;0.0;0.0;0.0;0.0;0.0]

for j = M:-1:1 

    din_now = zeros(6)
    din_old = zeros(6)

    autodiff(forward_func_4_AD, Duplicated(in_now[j], din_now), 
            Duplicated(in_old[j], din_old), Duplicated(zeros(6), dout_old), 
            Duplicated(zeros(6), dout_now))
    
    dout_old = copy(din_old)
    dout_now = copy(din_now)

    if j == 1
        return din_now, din_old
    end

end 

end

# First we complete step one and run the forward model:

const M = 10000             ## Deciding on total number of forward steps to take

states_smooth, states_unsmooth = forward_func(copy([T̄; S̄]), copy([T̄; S̄]), 10*day, M);   

# Next, we pass all of our states to the AD function to get back to the desired derivative:

adjoint_now, adjoint_old = ad_calc(states_unsmooth, states_smooth, M);

# And we're done! We were interested in sensitivity to the initial salinity of box 
# two, which will live in what we've called `adjoint_old`. Checking this value we see

@show adjoint_old[5]

# As it stands this is just a number, but a good check that Enzyme has computed what we want
# is to approximate the derivative with a Taylor series. Specifically,
#
# ```math
# J(\mathbf{x}(0) + \varepsilon) \approx J(\mathbf{x}(0)) + 
# \varepsilon \frac{\partial J}{\partial \mathbf{x}(0)}
# ```
#
# and a simple rearrangement yields
#
# ```math
# \frac{\partial J}{\partial \mathbf{x}(0)} \approx 
# \frac{J(\mathbf{x}(0) + \varepsilon)  - J(\mathbf{x}(0))}{\varepsilon}
# ```
#
# Hopefully we see that the analytical values converge close to the one we 
# found with Enzyme:

## unperturbed final state
use_to_check = states_smooth[M+1]

## a loop to compute the perturbed final state
diffs = []
step_sizes = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
for eps in step_sizes
    new1 = T̄
    new2 = S̄ + ([0.0;eps;0.0])g/kg
    state_old = [new1; new2];     
    state_new = zeros(6).*[°C,°C,°C,g/kg,g/kg,g/kg]                             
    state_now = [T̄; S̄];

    for t = 1:M

        ρ_now = ρ_func(state_now)
        u_now = U_func(ρ_now)
        state_new = timestep_func(state_now, state_old, u_now, 10d)

        for j = 1:6
            state_now[j] = state_now[j] + robert_filter_coeff * (state_new[j] - 2 * state_now[j] + state_old[j])
        end 

        state_old = state_now 
        state_now = state_new 
    
    
    end 

    temp = (state_old[1] - use_to_check[1])/eps;
    push!(diffs, temp)

end 

# Then checking what we found the derivative to be analytically:

@show diffs

# which comes very close to our calculated value. We can go further and check the 
# percent difference to see 

@show abs.(diffs .- adjoint_old[5])./adjoint_old[5]

# and we get down to a percent difference on the order of ``1e^{-5}``, showing Enzyme calculated
# the correct derivative. Success! 

