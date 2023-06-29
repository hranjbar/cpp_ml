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
	std::ostream& operator<<(std::ostream& os, const std::vector<int64_t>& v);
}