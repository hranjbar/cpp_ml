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
	namespace models
	{
		class Model
		{
		public:
			virtual void summary() = 0;

		protected:
			std::string inputName_, outputName_;
			std::vector<std::size_t> inputDimensions_, outputDimensions_;
		};
	}
}

