# What Effects Fuel Delivery To An Engine?

There are two primary states in which an engine operates that determine 
fuel delivery.
These are open-loop (based on defined parameters) and 
closed-loop (based on conditions).
When an engine first starts and is cold, 
it is initially in its open-loop mode.
For the sake of fuel consumption, 
the goal is to get the engine operating in closed-loop mode.

In closed loop mode, the ECU will adjust the fuel delivered to the engine based 
on: the O<sup>2</sup> sensors, 
the Mass Air Flow (MAF) sensor and the Manifold Absolute Pressure (MAP) sensor,
the coolant temperature sensor (for entering this mode mostly),
engine load, engine speed, and throttle position.
There are also various aspects of the fuel system, 
along with a fuel trim that has a more minor effect on fuel consumption.
This allows the ECU to facilitate the efficient operation of the engine,
along with prolonging its life.

Within our data, we have the outdoor temperature, intake air temperature (MAF),
engine load, engine speed, and throttle position. 
While these are not enough to calculate the exact fuel consumption, 
they are enough to get an approximate value for the sake of comparison.
Furthermore, there are more than just these factors that affect fuel 
consumption like speed, acceleration, gear selection, accessory load, 
torque converter state, and so on.
