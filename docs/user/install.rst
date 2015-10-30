Installation
=============

Dependencies
------------

This package depends on several libraries (the number varies according to which modules you want to use).

**Basic dependencies**
	Regardless of your needs, the following libraries are required:
		* numpy
		* scipy
		* graph_tool 

	If you want to simulate activities on your complex networks, AGNet can directly interact with the NEST simulator to implement the network inside PyNEST. For this, you will need to install NEST with Python bindings, which requires:
		* Python headers (`python-dev` package on most linux distributions)
		* autoconf
		* automake
		* libtool
		* libltdl
