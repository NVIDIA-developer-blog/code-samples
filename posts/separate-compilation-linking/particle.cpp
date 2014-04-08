#include <particle.h>

particle::particle() : 	position(), velocity(), totalDistance(0,0,0)
{}

__device__ __host__ 
void particle::advance(float d)
{
	velocity.normalize();
	float dx = d * velocity.x;
	position.x += dx;
	totalDistance.x += dx;
	float dy = d * velocity.y;
 	position.y += dy;
 	totalDistance.y += dy;
 	float dz = d * velocity.z;
	position.z += dz;
	totalDistance.z += dz;
	velocity.scramble();
}

const v3& particle::getTotalDistance() const
{	return totalDistance; }