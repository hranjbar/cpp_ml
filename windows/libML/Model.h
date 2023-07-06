/*---------------------------------------------------------------------------*/
/*        Copyright (c) Siemens Medical Solutions USA, Inc.			         */
/*                            All Rights Reserved                            */
/*---------------------------------------------------------------------------*/
/*
	note:		Abstract class for machine learning (ML) models
	author:		Homayoon Ranjbar (homayoon.ranjbar@siemens-healthineers.com)
	date:		July 2023
*/

#pragma once

#include <string>
#include <vector>

namespace ml
{
	class Model
	{
	public:
		virtual void summary() = 0;

	protected:
		std::string inputName_, outputName_;
		std::vector<int64_t> inputDimensions_, outputDimensions_;
	};
}

