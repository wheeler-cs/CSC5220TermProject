"""
Make all the figures
"""
import multiprocessing
import time
from make_figures.altitude_vs_mpg import make_altitude_plot
from make_figures.bearing_vs_mpg import make_bearing_mpg_plot
from make_figures.g_vs_mpg import make_g_mpg_plot
from make_figures.grade_vs_mpg import make_grade_mpg_plot
from make_figures.histograms import make_histograms
from make_figures.intake_air_temp_mpg import make_intake_air_temp_mpg
from make_figures.mpg_rpm_speed_linear_model import make_mpg_rpm_linear_model
from make_figures.mpg_speed_linear_model import make_mpg_linear_model
from make_figures.mpg_weather_speed_linear_model import make_mpg_weather_linear_model
from make_figures.rpm_vs_mpg import make_rpm_mpg_plot
from make_figures.speed_vs_mpg import make_speed_mpg_plot
from make_figures.temp_speed_vs_mpg import make_temp_speed_mpg
from make_figures.throttle_vs_mpg import make_throttle_mpg_plot
from make_figures.weather_mpg import make_weather_mpg, make_weather_mpg_fahrenheit
from make_figures.weather_versus_intake_air import make_weather_intake_air_fahrenheit

if __name__ == '__main__':
    start = time.perf_counter()
    functions = [
        make_altitude_plot,
        make_bearing_mpg_plot,
        make_grade_mpg_plot,
        make_intake_air_temp_mpg,
        make_mpg_rpm_linear_model,
        make_mpg_linear_model,
        make_mpg_weather_linear_model,
        make_rpm_mpg_plot,
        make_speed_mpg_plot,
        make_temp_speed_mpg,
        make_throttle_mpg_plot,
        make_weather_mpg,
        make_weather_mpg_fahrenheit,
        make_weather_intake_air_fahrenheit
    ]

    # Make the figures
    # pylint: disable=consider-using-with
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # Add the single-thread plots
    for func in functions:
        pool.apply_async(func)

    # Add plots that use the pool
    make_g_mpg_plot(pool)
    make_histograms(pool)

    pool.close()
    pool.join()
    end = time.perf_counter()
    print(f"Took {end - start:.2f}s to make figures")
