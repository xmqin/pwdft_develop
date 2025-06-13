/// @file global.cpp
/// @brief Global structure.
/// @date 2023-07-01
#include    "environment.hpp"

namespace PPCG{
// *********************************************************************
// IO
// *********************************************************************
std::ofstream  statusOFS;

// *********************************************************************
// Error handling
// *********************************************************************
void ErrorHandling( const char * msg ){
  statusOFS << std::endl << "ERROR!" << std::endl 
    << msg << std::endl << std::endl;
  throw std::runtime_error( msg );
}

} // namespace PPCG
