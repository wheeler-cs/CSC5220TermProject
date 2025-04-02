"""
Make all the plots
"""
from make_figures.altitude_vs_mpg import make_altitude_plot
from make_figures.histograms import make_histograms
from make_figures.intake_air_temp_mpg import make_intake_air_temp_mpg
from make_figures.mpg_rpm_speed_linear_model import make_mpg_rpm_linear_model
from make_figures.mpg_speed_linear_model import make_mpg_linear_model
from make_figures.speed_vs_mpg import make_speed_mpg_plot
from make_figures.temp_speed_vs_mpg import make_temp_speed_mpg
from make_figures.weather_mpg import make_weather_mpg, make_weather_mpg_fahrenheit


if __name__ == '__main__':
    make_altitude_plot()
    make_histograms()
    make_intake_air_temp_mpg()
    make_mpg_rpm_linear_model()
    make_mpg_linear_model()
    make_speed_mpg_plot()
    make_temp_speed_mpg()
    make_weather_mpg()
    make_weather_mpg_fahrenheit()
