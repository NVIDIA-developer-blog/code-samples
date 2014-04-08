#ifndef __v3_h__
#define __v3_h__

class v3
{
public:
	float x;
	float y;
	float z;
	
	v3();
	v3(float xIn, float yIn, float zIn);
	void randomize();
	__host__ __device__ void normalize();
	__host__ __device__ void scramble();

};

#endif