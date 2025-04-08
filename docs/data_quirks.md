# Data Quirks

There are some quirks in the data that will lead to some inaccuracies in any 
model that isn't severely overfit.

1. Torque converter lockup

    - This is based on a variety of conditions and *may* involve some amount of
      driver and system learning.

    - It can be coaxed into happening depending on the speed, gear, 
      and acceleration currently being experienced. This can be used to slow 
      down or maintain speed, depending on the grade (angle) of the road.

      - This usually involves using the gas pedal, so I'm using the gas pedal 
        to slow down. Backward sounding, but true. Light application of the 
        brakes can also do this by also coaxing a downshift.

    - It means that engine RPM and speed aren't always directly correlated like
      they would be in a vehicle with a manual transmission. The engine could be
      at >2k RPM at 55 when it's unlocked in fifth gear, but at ~2k RPM at 
      62 MPH when it's locked and also in fifth gear.

    - Torque converter lockup is the most efficient state of an automatic 
      transmission for a particular gear. Torque converters use a fluid to 
      translate speed into greater torque, hence their name. This is not an
      efficient process, but allows for more torque at a given speed in a 
      given gear. 

2. Fuel energy content

    - Fuel has differing energy content between summer and winter blends,
      though this is a more minor aspect of the data with the proxy of outdoor
      temperature. 
    
      - With how long I go between fills, this gets further spread out.

3. Vehicle state of repair

    - With the time span of the data, it includes various states of repair.

        - Though, this is mostly oil changes and tire pressure.

    - I did buy a new set of tires somewhere in this data which lead to better
      fuel consumption to an extent. This is one of those many things that will
      add up over time.

    - I need a new VTEC spool valve or at least a seal for it.

    - I need axles eventually. Just a high milage thing.

    - My AC isn't in the greatest state of repair, so that's doing something.

      - The relay for the compressor's magnetic clutch went out at some point 
        in the data, which was a short-lived increase.

      - The condenser freezes up after about an hour and a half to two hours of 
        driving with it at or near full blast. So, I don't do that much.

    - My brakes are a bit rusty at times due to how much I drive, so some data 
      includes excessive braking to get rid of that.

    - I lost a mudflap somewhere at some point in time. No idea when or where.

      - This has a minor effect on aerodynamics.

4. VTEC

    - VTEC stands for Variable Valve Timing and Lift Electronic Control. 
      It is a system developed by Honda to optimize the engine's performance 
      at different RPMs by adjusting the timing and lift of the engine's intake 
      and exhaust valves.

    - My car specifically has a form of iVTEC that is backward to the normal
      VTEC systems you may know of. The "performance" cam is the normal cam
      for idling and acceleration, and the "economy" cam is only employed for
      light engine load driving between 1000 and 3000 RPM.

      - This means I'm driving in VTEC a lot of the time, instead of rarely.

    - The "economy" cam straddles the line between the Atkinson cycle and 
      the Otto cycle by delaying the intake valve closing by a small amount,
      but not enough to be considered an Atkinson cycle.
    
    - I don't have an indication of any sort for which cam is being used, so it
      is not present in the data. Theoretically, it should be a relatively 
      smooth transition when plotted as a surface plot with the correct factors.

    - If you're curious enough, here's a [video](https://youtu.be/QbONaEK08as?si=PX7zimbNyfsXcwPQ).

    - Obligatory [video](https://www.youtube.com/watch?v=aAZLarnkNuE)

5. Weather

    - My driving habits are affected by rain and especially snow.

        - When it rains, I drive more like a normal person. Basically, I follow 
          the speed limit a bit closer.
    
        - When it snows, I take it easy and try to not let my foot slip. 
          Overall, I drive like I'm in a Civic on all-season tires in a
          less-than-ideal traction situation. This includes minimal braking so
          that my backend stays following and isn't trying to lead.
        
        - When the weather is clear, I tend to drive at roughly the speed limit,
          not using my brakes excessively. 

6. Driving Style

    - My driving style has some predictable aspects of it.
   
    - DWL (Driving With Load)
    
      - Increase speed down hills, decrease speed going up hills.
      
    - I don't like to "hurry up and stop." (rabbit driving)

      - I'll take my foot off the gas pedal as I approach a stop, 
        hitting my brakes as needed when I get closer.

      - Traffic dictates the time span which this occurs in. I might not 
        even hit the speed limit if the next stop is too close. 
    
    - Some of the roads I drive on have speed limits that aren't comfortable to
      hit for large portions of the road.

        - They're mostly curvy roads with speed limits you'll only hit on the 
          straighter portions.

        - I could more consistently hit or exceed the speed limit on these roads,
          but I don't like to unless I'm bored. Some people on these roads will 
          get impatient and pass you over a double-yellow line.
    
    - Pulse and glide

        - For some roads, it makes sense to speed up (pulse) and coast (glide).
          When done right, this is more efficient than maintaining a constant
          speed. Usually, I just do this on roads where it makes sense to have
          this variability of speed in the first place. See DFCO later in this.

    - Traffic

        - Traffic will, of course, impact my fuel consumption based on things 
          like speed, stopping time, standstills, etc.

        - I tend to drive more like a semi-truck driver does in traffic, in that
          I try to approximate the speed of the flow more than stopping and 
          starting with the flow of said traffic. 

    - I did a lot of learning on a manual transmission.

      - Some of my driving habits come from driving a manual transmission, so I
        do things like downshift for long hills, coax downshifts as I approach
        a stop, manually select 2<sup>nd</sup> gear for particularly slow school
        zones, and so on.

      - Automatic transmissions annoy me, because they like to be in the wrong
        gear (to me) because they can't see the road ahead.

      - Aside from the increase in mechanical efficiency, I am capable of 
        selecting the correct gear better than my car is for efficiency or 
        power. I can also do so as smoothly or more so than my transmission if
        I actually try. Mind you, my car is an automatic, so I cannot do this.
        I have driven almost the same car as mine, but manual, so I know I
        can drive the manual equivalent like this.

    - I don't use cruise control

      - It's less efficient for me due to other driving methods.

      - I have a light foot most of the time, so my exceeding of the speed 
        limit is going to be more efficient than keeping a stead speed of the
        speed limit.

7. DFCO: Deceleration Fuel Cutoff

    - This is a part of OBD-II vehicles for emissions. When the vehicle is 
      coasting in gear with the torque converter locked up (clutch in for 
      manual transmissions), the ECU (Engine Control Unit) cuts fuel to the
      engine. This is why you will see places in the data with zero fuel 
      used and an instant MPG of 512 (the Torque app's equivalent of infinite 
      MPG).

    - I use this going down hills, when coming to a stop, when I need to 
      slow down, or to maintain a speed.
