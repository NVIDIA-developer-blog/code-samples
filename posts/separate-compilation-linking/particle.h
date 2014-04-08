#ifndef __particle_h__
#define __particle_h__

#include <v3.h>

class particle
{
	private:
		v3 position;
		v3 velocity;
		v3 totalDistance;

	public:
		particle();
		__host__ __device__ void advance(float dist);
		const v3& getTotalDistance() const;

};

#endif