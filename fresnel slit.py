import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

#Defines system parameters.
D=5000
d=100
wavelen=0.5
L=5000
x=np.linspace(-L/2,L/2,int(L))

#Define the slit aperture function.
def A(x,d):
	return (np.s(x)<d/2).astype(float)

#Define modified aperture function as required for fresnel diffraction.
def Aprime(x,d):
	return np.exp(1j*np.pi*np.power(x,2)/(wavelen*D))*A(x,d)

#Perform fresnel diffraction fourier integral.
def intensity(x,wavelen,D,d):
	ft=np.fft.fft(Aprime(x,d))
	spatfreq=np.fft.fftfreq(len(x),x[1]-x[0])
	y=spatfreq*wavelen*D
	amp=np.exp(1j*2*np.pi/wavelen*np.power(y,2)/(2*D))*ft
	return np.vstack((y,np.power(np.abs(amp)/(np.abs(amp[0])),2),np.angle(amp)))

pattern=intensity(x,wavelen,D,d)
#Computes what the fraunhofer result would be: so that we see the fraunhofer approximation gives invalid results.
pred=np.power(np.sinc(d*pattern[0,:]/(D*wavelen)),2)


#Plot the results.
fig, ax=plt.subplots()
simulation, = ax.plot(pattern[0,:],pattern[1,:])
theory, = ax.plot(pattern[0,:],pred)
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_position('center')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xlabel("screen position/um",x=0.5)
ax.xaxis.set_label_position('bottom') 
ax.set_ylabel("normalised intensity/arbitrary",y=0.5)
ax.legend([simulation, theory], ["fft prediction", "fraunhofer result"])
plt.ylim((0,1))
plt.xlim((-500,500))
plt.savefig("fresnel slit.png",dpi=200)


