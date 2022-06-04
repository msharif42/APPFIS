/**
 *
 * Header to include for using the framework
 * Author: Md Bulbul Sharif
 *
 **/


#ifndef APPFIS_HPP
#define APPFIS_HPP


#include "Env.hpp"
#include "Common.hpp"
#include "Grid.hpp"
#include "Runtime.hpp"

#ifdef APPFIS_CUDA
#include "Runtime_CUDA.hpp"
#endif

#include "Communication.hpp"
#include "Timer.hpp"


#endif