import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


#------------
#  data
#------------
dirname = os.path.dirname(__file__)
time_filename = os.path.join(dirname, '../data/coronavirusdataset_ver2/Time.csv')
time_age_filename = os.path.join(dirname, '../data/coronavirusdataset_ver2/TimeAge.csv')
time_gender_filename = os.path.join(dirname, '../data/coronavirusdataset_ver2/TimeGender.csv')
time_province_filename = os.path.join(dirname, '../data/coronavirusdataset_ver2/TimeProvince.csv')
weather_filename = os.path.join(dirname, '../data/coronavirusdataset_ver2/Weather.csv')


# ----------------------------------
#  read data & prepare dataframe
# ----------------------------------
test_data = pd.read_csv(time_filename)
test_data['day_after_first_outbreak'] = test_data.index + 1   # day = index + 1

gender_data = pd.read_csv(time_gender_filename)
province_data = pd.read_csv(time_province_filename)
temperature_data = pd.read_csv(weather_filename)
seoul_weather_data = temperature_data.loc[temperature_data['province'] == 'Seoul']
busan_weather_data = temperature_data.loc[temperature_data['province'] == 'Busan']


def show_numtest_positive_plot(data):
  N = len(data)   # number of date

  accum_num_test = data['test']
  accum_num_positive = data['confirmed']
  ind = np.arange(N)  # the x locations for the groups
  width = 0.35  # the width of the bars: can also be len(x) sequence

  plt.figure(figsize=(8,10))
  p1 = plt.bar(ind, accum_num_positive, width)
  p2 = plt.bar(ind, accum_num_test, width)

  #plt.yscale('log')
  plt.ylabel('Accumulated COVID-19 Tests')
  plt.title('Accumulated number of COVID-19 Tests and Positive cases \n since first outbreak (Jan 20. 2020)')
  plt.xticks(ind, data['date'], rotation=90, fontsize=6)
  plt.legend((p1[0], p2[0]), ('number of positive cases', 'number of tests'))

  def autolabel(rects):
    for rect in rects:
      height = rect.get_height()
      plt.annotate('{}'.format(height),
                  xy=(rect.get_x() + rect.get_width() / 2, height),
                  xytext=(0, 5),  # 3 points vertical offset
                  textcoords="offset points",
                  ha='center', va='bottom', fontsize=5)

  autolabel(p2)
  plt.show()


def gender_cases_histogram(data):
  male_data = data.loc[data['sex'] == 'male']
  female_data = data.loc[data['sex'] == 'female']

  x = np.arange(len(male_data))
  width = 0.3  # the width of the bars

  fig, ax = plt.subplots()
  rects1 = ax.bar(x - width / 2, male_data['confirmed'], width, label='Men')
  rects2 = ax.bar(x + width / 2, female_data['confirmed'], width, label='Women')

  ax.set_ylabel('Number of positive cases')
  ax.set_title('Accumulated number of COVID-19 positive cases per gender')
  ax.set_xticks(x)
  ax.set_xticklabels(male_data['date'], fontsize=7)
  ax.legend()

  def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
      height = rect.get_height()
      ax.annotate('{}'.format(height),
                  xy=(rect.get_x() + rect.get_width() / 2, height),
                  xytext=(0, 3),  # 3 points vertical offset
                  textcoords="offset points",
                  ha='center', va='bottom', fontsize=6)

  autolabel(rects1)
  autolabel(rects2)

  fig.tight_layout()

  plt.show()


def deceased_cases_histogram(data):
  male_data = data.loc[data['sex'] == 'male']
  female_data = data.loc[data['sex'] == 'female']

  x = np.arange(len(male_data))
  width = 0.3  # the width of the bars

  fig, ax = plt.subplots()
  rects1 = ax.bar(x - width / 2, male_data['deceased'], width, label='Men')
  rects2 = ax.bar(x + width / 2, female_data['deceased'], width, label='Women')

  ax.set_ylabel('Number of deceased')
  ax.set_title('Accumulated number of deceased by COVID-19 per gender')
  ax.set_xticks(x)
  ax.set_xticklabels(male_data['date'], fontsize=8)
  ax.legend()

  def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
      height = rect.get_height()
      ax.annotate('{}'.format(height),
                  xy=(rect.get_x() + rect.get_width() / 2, height),
                  xytext=(0, 3),  # 3 points vertical offset
                  textcoords="offset points",
                  ha='center', va='bottom')

  autolabel(rects1)
  autolabel(rects2)
  fig.tight_layout()
  plt.show()


def get_confirmed_cases_seoul_and_weather(province_data):
  seoul_data = province_data.loc[province_data['province'] == 'Seoul']
  x = np.arange(len(seoul_data))
  width = 0.5  # the width of the bars

  fig, ax = plt.subplots()
  rects1 = ax.bar(x - width / 2, seoul_data['confirmed'], width, label='confirmed')

  ax.set_ylabel('Number of positive cases in Seoul')
  ax.set_title('Accumulated number of COVID-19 positive cases in Seoul')
  ax.set_xticks(x)  # how to rotate x label
  ax.set_xticklabels(seoul_data['date'], fontsize=7)
  ax.legend()

  def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
      height = rect.get_height()
      ax.annotate('{}'.format(height),
                  xy=(rect.get_x() + rect.get_width() / 2, height),
                  xytext=(0, 3),  # 3 points vertical offset
                  textcoords="offset points",
                  ha='center', va='bottom')

  autolabel(rects1)
  fig.tight_layout()

  plt.show()


def get_confirmed_cases_pusan_and_weather(province_data):
  busan_data = province_data.loc[province_data['province'] == 'Busan']
  x = np.arange(len(busan_data))
  width = 0.5  # the width of the bars

  fig, ax = plt.subplots()
  rects1 = ax.bar(x - width / 2, busan_data['confirmed'], width, label='confirmed')

  ax.set_ylabel('Number of Positive cases in Busan')
  ax.set_title('Accumulated Number of COVID-19 positive cases in Busan')
  ax.set_xticks(x)
  ax.xticks(busan_data['date'],rotation=90,fontsize=7)
  ax.legend()

  def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
      height = rect.get_height()
      ax.annotate('{}'.format(height),
                  xy=(rect.get_x() + rect.get_width() / 2, height),
                  xytext=(0, 3),  # 3 points vertical offset
                  textcoords="offset points",
                  ha='center', va='bottom')

  autolabel(rects1)
  fig.tight_layout()
  plt.show()


def seoul_positive_cases_weather(data):
  print(data)
  import matplotlib.pyplot as plt
  import numpy as np

  plt.clf()
  xs = data['date']
  ys_bars = data['confirmed']
  ys_lines = data['avg_temp']

  N = len(data)
  ind = np.arange(N)

  # this is the axis on the left side of chart
  ax1 = plt.gca()
  ax1.bar(xs, ys_bars, color='green')

  # Ticks must be set after the plot has been drawn
  ax1.set_yticks(np.arange(0, 280, 10))
  ax1.set_yticklabels(np.arange(0, 280, 10), color='green')

  # create the 'twin' axis on the right
  ax2 = ax1.twinx()

  ax1.set_ylabel('Number of Positive cases')
  ax1.set_title('Accumulated COVID-19 positive cases in Seoul \n and Average Temperature')
  ax1.set_xticklabels(data['date'], rotation=90, fontsize=6)
  ax1.legend()

  ax2.plot(xs, ys_lines, '--', color='red')
  ax2.set_ylabel('Average temperature (Celsius)')
  ax2.set_yticks(np.arange(0, 50, 5))
  ax2.set_yticklabels(np.arange(0, 50, 5), color='red')
  ax2.tick_params(axis='x', rotation=45)

  data['day_after_first_outbreak'] = data.index + 1
  ind = np.arange(len(data))
  plt.xticks(ind, data['date'])
  plt.show()


def busan_positive_cases_weather(data):
  import matplotlib.pyplot as plt
  import numpy as np

  plt.clf()

  xs = data['date']
  ys_bars = data['confirmed']
  ys_lines = data['avg_temp']

  ax1 = plt.gca()
  ax1.bar(xs, ys_bars, color='green')
  ax1.set_yticks(np.arange(0, 280, 10))
  ax1.set_yticklabels(np.arange(0, 280, 10), color='green')

  # create the 'twin' axis on the right
  ax2 = ax1.twinx()

  ax1.set_ylabel('Number of Positive cases')
  ax1.set_title('Accumulated COVID-19 positive cases in Busan \n and Average Temperature')
  ax1.set_xticklabels(data['date'], rotation=90, fontsize=6)
  ax1.legend()

  ax2.plot(xs, ys_lines, '--', color='red')
  ax2.set_ylabel('Average temperature (Celsius)')
  ax2.set_yticks(np.arange(0, 101, 10))
  ax2.set_yticklabels(np.arange(0, 101, 10), color='red')
  ax2.xaxis.set_ticks(xs)
  ax2.tick_params(axis='x', rotation=45)

  data['day_after_first_outbreak'] = data.index + 1
  ind = np.arange(len(data))
  plt.xticks(ind, data['date'], rotation=90)

  plt.show()



# -----------------------
#  plot data
# ----------------------

show_numtest_positive_plot(test_data)

gender_cases_histogram(gender_data)

deceased_cases_histogram(gender_data)

seoul_data = province_data.loc[province_data['province'] == 'Seoul']
weather_in_seoul = seoul_weather_data
merged_seoul = pd.merge(seoul_data, weather_in_seoul, left_on='date', right_on='date')
seoul_positive_cases_weather(merged_seoul)

busan_data = province_data.loc[province_data['province'] == 'Busan']
weather_in_busan = busan_weather_data
merged_busan = pd.merge(busan_data, weather_in_busan, left_on='date', right_on='date')
busan_positive_cases_weather(merged_busan)

