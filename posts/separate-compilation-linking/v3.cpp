#include <v3.h>
#include <math.h>

v3::v3()
{	randomize(); }

v3::v3(float xIn, float yIn, float zIn) : x(xIn), y(yIn), z(zIn)
{}

void v3::randomize()
{
	x = (float)rand() / (float)RAND_MAX;
	y = (float)rand() / (float)RAND_MAX;
	z = (float)rand() / (float)RAND_MAX;
}

__host__ __device__ void v3::normalize()
{
	float t = sqrt(x*x + y*y + z*z);
	x /= t;
	y /= t;
	z /= t;
}

__host__ __device__ void v3::scramble()
{
	float tx = 0.317f*(x + 1.0) + y + z * x * x + y + z;
	float ty = 0.619f*(y + 1.0) + y * y + x * y * z + y + x;
	float tz = 0.124f*(z + 1.0) + z * y + x * y * z + y + x;
	x = tx;
	y = ty;
	z = tz;
}
