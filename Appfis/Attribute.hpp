/**
 *
 * Structure to contain all customize initial parameter
 * Author: Md Bulbul Sharif
 *
 **/


#ifndef ATTRIBUTE_HPP
#define ATTRIBUTE_HPP


namespace APPFIS
{

	struct Attribute {

		Attribute() : ghostLayer(1), threads(1), periodic(false), overlap(false) {}
		int ghostLayer;
		int threads;
		bool periodic;
		bool overlap;

	} ATTRIBUTE;

}


#endif