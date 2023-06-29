/*---------------------------------------------------------------------------*/
/*        Copyright (c) Siemens Medical Solutions USA, Inc.			         */
/*                            All Rights Reserved                            */
/*---------------------------------------------------------------------------*/
/*
	author:		Homayoon Ranjbar (homayoon.ranjbar@siemens-healthineers.com)
	date:		June 2023
*/

#pragma once
#include <iostream>
#include <vector>

namespace ml
{
		template<typename T>
		std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) 
		{
			os << "[";
			for (int i = 0; i < v.size() - 1; ++i) os << v[i] << ", ";
			os << v.back() << "]";
			return os;
		}
}