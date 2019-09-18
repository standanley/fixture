// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header

#ifndef _VMyAmpInterface__Syms_H_
#define _VMyAmpInterface__Syms_H_

#include "verilated.h"

// INCLUDE MODULE CLASSES
#include "VMyAmpInterface.h"

// SYMS CLASS
class VMyAmpInterface__Syms : public VerilatedSyms {
  public:
    
    // LOCAL STATE
    const char* __Vm_namep;
    bool __Vm_didInit;
    
    // SUBCELL STATE
    VMyAmpInterface*               TOPp;
    
    // CREATORS
    VMyAmpInterface__Syms(VMyAmpInterface* topp, const char* namep);
    ~VMyAmpInterface__Syms() {}
    
    // METHODS
    inline const char* name() { return __Vm_namep; }
    
} VL_ATTR_ALIGNED(64);

#endif // guard
