import soundfile
import numpy as np
import functions as my_f

#%%

s_back, sr_back = soundfile.read('background.wav', dtype='float32')
s_vessel, sr_vessel = soundfile.read('vessel.wav', dtype='float32')

s_back = my_f.standardization(s_back)
s_vessel = my_f.standardization(s_vessel)

my_f.plot_signal(s_back, sr=sr_back, name='Back-all')
my_f.plot_signal(s_vessel, sr=sr_vessel, name='Vessel-all')

si = sr_back*10
sf = si + sr_back*1
sb = s_back[si:sf]
sv = s_vessel[si:sf]
print(f'{si} {sf}')

sb = my_f.standardization(sb)
sv = my_f.standardization(sv)

my_f.save_audio(sb, sr=sr_back, name='back_cut')
my_f.save_audio(sv, sr=sr_vessel, name='vessel_cut')

## back

my_f.plot_signal(sb, sr=sr_back, name='Back')
my_f.spec_plot(sb, ms=23, sr=sr_back, name='Back')
sc_back, dc_back, imfs_b = my_f.get_components(sb, cutoff=0.025)
my_f.spec_plot(dc_back, ms=23, sr=sr_back, name='dc_Back')
my_f.spec_plot(sc_back, ms=23, sr=sr_back, name='sc_Back')
my_f.save_audio(dc_back, sr=sr_vessel, name='dc_back')
my_f.save_audio(sc_back, sr=sr_vessel, name='sc_back')

## vessel
my_f.plot_signal(sv, sr=sr_back, name='Vessel')
my_f.spec_plot(sv, ms=23, sr=sr_vessel, name='Vessel')
sc_vessel, dc_vessel, imfs_v = my_f.get_components(sv, cutoff=0.025)
my_f.spec_plot(dc_vessel, ms=23, sr=sr_back, name='dc_Vessel')
my_f.spec_plot(sc_vessel, ms=23, sr=sr_back, name='sc_Vessel')
my_f.save_audio(dc_vessel, sr=sr_vessel, name='dc_v')
my_f.save_audio(sc_vessel, sr=sr_vessel, name='sc_v')



ns = my_f.add_white_noise(sv, alpha=0.1)
my_f.plot_signal(ns, sr=sr_back, name='noise')
my_f.save_audio(ns, sr=sr_vessel, name='noise')

## others
dc_sum = dc_vessel + dc_back
dc_sum = my_f.standardization(dc_sum)
my_f.plot_signal(dc_sum, sr=sr_back, name='dc_sum')
my_f.save_audio(dc_sum, sr=sr_vessel, name='dc_sum')

dc_sum = dc_vessel + sc_back
dc_sum = my_f.standardization(dc_sum)
my_f.plot_signal(dc_sum, sr=sr_back, name='dc_sum')
my_f.save_audio(dc_sum, sr=sr_vessel, name='dc_sum')


## ICA
a = 8000*5
b = a + 8000*30
X = list(zip(s_vessel[a:b], s_back[a:b]))
# Import FastICA
from sklearn.decomposition import FastICA

# Initialize FastICA with n_components=3
ica = FastICA(n_components=2)

# Run the FastICA algorithm using fit_transform on dataset X
ica_result = ica.fit_transform(X)

result_signal_1 = ica_result[:,0]
result_signal_2 = ica_result[:,1]
result_signal_1 = my_f.standardization(result_signal_1)
result_signal_2 = my_f.standardization(result_signal_2)
my_f.plot_signal(result_signal_1, sr=sr_back, name='result_signal_1')
my_f.plot_signal(result_signal_2, sr=sr_back, name='result_signal_2')
my_f.save_audio(result_signal_1, sr=sr_vessel, name='result_signal_1')
my_f.save_audio(result_signal_2, sr=sr_vessel, name='result_signal_2')
