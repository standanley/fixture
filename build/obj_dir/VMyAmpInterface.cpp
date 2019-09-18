// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VMyAmpInterface.h for the primary calling header

#include "VMyAmpInterface.h"   // For This
#include "VMyAmpInterface__Syms.h"


//--------------------
// STATIC VARIABLES


//--------------------

VL_CTOR_IMP(VMyAmpInterface) {
    VMyAmpInterface__Syms* __restrict vlSymsp = __VlSymsp = new VMyAmpInterface__Syms(this, name());
    VMyAmpInterface* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Reset internal values
    
    // Reset structure values
    _ctor_var_reset();
}

void VMyAmpInterface::__Vconfigure(VMyAmpInterface__Syms* vlSymsp, bool first) {
    if (0 && first) {}  // Prevent unused
    this->__VlSymsp = vlSymsp;
}

VMyAmpInterface::~VMyAmpInterface() {
    delete __VlSymsp; __VlSymsp=NULL;
}

//--------------------


void VMyAmpInterface::eval() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate VMyAmpInterface::eval\n"); );
    VMyAmpInterface__Syms* __restrict vlSymsp = this->__VlSymsp;  // Setup global symbol table
    VMyAmpInterface* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
#ifdef VL_DEBUG
    // Debug assertions
    _eval_debug_assertions();
#endif // VL_DEBUG
    // Initialize
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) _eval_initial_loop(vlSymsp);
    // Evaluate till stable
    int __VclockLoop = 0;
    QData __Vchange = 1;
    while (VL_LIKELY(__Vchange)) {
	VL_DEBUG_IF(VL_DBG_MSGF("+ Clock loop\n"););
	_eval(vlSymsp);
	__Vchange = _change_request(vlSymsp);
	if (VL_UNLIKELY(++__VclockLoop > 100)) VL_FATAL_MT(__FILE__,__LINE__,__FILE__,"Verilated model didn't converge");
    }
}

void VMyAmpInterface::_eval_initial_loop(VMyAmpInterface__Syms* __restrict vlSymsp) {
    vlSymsp->__Vm_didInit = true;
    _eval_initial(vlSymsp);
    int __VclockLoop = 0;
    QData __Vchange = 1;
    while (VL_LIKELY(__Vchange)) {
	_eval_settle(vlSymsp);
	_eval(vlSymsp);
	__Vchange = _change_request(vlSymsp);
	if (VL_UNLIKELY(++__VclockLoop > 100)) VL_FATAL_MT(__FILE__,__LINE__,__FILE__,"Verilated model didn't DC converge");
    }
}

//--------------------
// Internal Methods

VL_INLINE_OPT void VMyAmpInterface::_combo__TOP__1(VMyAmpInterface__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VMyAmpInterface::_combo__TOP__1\n"); );
    VMyAmpInterface* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->my_out = (1U & (~ (IData)(vlTOPp->my_in)));
}

void VMyAmpInterface::_eval(VMyAmpInterface__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VMyAmpInterface::_eval\n"); );
    VMyAmpInterface* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->_combo__TOP__1(vlSymsp);
}

void VMyAmpInterface::_eval_initial(VMyAmpInterface__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VMyAmpInterface::_eval_initial\n"); );
    VMyAmpInterface* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void VMyAmpInterface::final() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VMyAmpInterface::final\n"); );
    // Variables
    VMyAmpInterface__Syms* __restrict vlSymsp = this->__VlSymsp;
    VMyAmpInterface* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void VMyAmpInterface::_eval_settle(VMyAmpInterface__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VMyAmpInterface::_eval_settle\n"); );
    VMyAmpInterface* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->_combo__TOP__1(vlSymsp);
}

VL_INLINE_OPT QData VMyAmpInterface::_change_request(VMyAmpInterface__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VMyAmpInterface::_change_request\n"); );
    VMyAmpInterface* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // Change detection
    QData __req = false;  // Logically a bool
    return __req;
}

#ifdef VL_DEBUG
void VMyAmpInterface::_eval_debug_assertions() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VMyAmpInterface::_eval_debug_assertions\n"); );
    // Body
    if (VL_UNLIKELY((my_in & 0xfeU))) {
	Verilated::overWidthError("my_in");}
}
#endif // VL_DEBUG

void VMyAmpInterface::_ctor_var_reset() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VMyAmpInterface::_ctor_var_reset\n"); );
    // Body
    my_in = VL_RAND_RESET_I(1);
    my_out = VL_RAND_RESET_I(1);
}
