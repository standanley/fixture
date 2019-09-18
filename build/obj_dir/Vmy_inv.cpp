// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vmy_inv.h for the primary calling header

#include "Vmy_inv.h"           // For This
#include "Vmy_inv__Syms.h"


//--------------------
// STATIC VARIABLES


//--------------------

VL_CTOR_IMP(Vmy_inv) {
    Vmy_inv__Syms* __restrict vlSymsp = __VlSymsp = new Vmy_inv__Syms(this, name());
    Vmy_inv* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Reset internal values
    
    // Reset structure values
    _ctor_var_reset();
}

void Vmy_inv::__Vconfigure(Vmy_inv__Syms* vlSymsp, bool first) {
    if (0 && first) {}  // Prevent unused
    this->__VlSymsp = vlSymsp;
}

Vmy_inv::~Vmy_inv() {
    delete __VlSymsp; __VlSymsp=NULL;
}

//--------------------


void Vmy_inv::eval() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vmy_inv::eval\n"); );
    Vmy_inv__Syms* __restrict vlSymsp = this->__VlSymsp;  // Setup global symbol table
    Vmy_inv* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
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

void Vmy_inv::_eval_initial_loop(Vmy_inv__Syms* __restrict vlSymsp) {
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

VL_INLINE_OPT void Vmy_inv::_combo__TOP__1(Vmy_inv__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmy_inv::_combo__TOP__1\n"); );
    Vmy_inv* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->out = vlTOPp->in_;
}

void Vmy_inv::_eval(Vmy_inv__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmy_inv::_eval\n"); );
    Vmy_inv* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->_combo__TOP__1(vlSymsp);
}

void Vmy_inv::_eval_initial(Vmy_inv__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmy_inv::_eval_initial\n"); );
    Vmy_inv* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void Vmy_inv::final() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmy_inv::final\n"); );
    // Variables
    Vmy_inv__Syms* __restrict vlSymsp = this->__VlSymsp;
    Vmy_inv* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void Vmy_inv::_eval_settle(Vmy_inv__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmy_inv::_eval_settle\n"); );
    Vmy_inv* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->_combo__TOP__1(vlSymsp);
}

VL_INLINE_OPT QData Vmy_inv::_change_request(Vmy_inv__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmy_inv::_change_request\n"); );
    Vmy_inv* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // Change detection
    QData __req = false;  // Logically a bool
    return __req;
}

#ifdef VL_DEBUG
void Vmy_inv::_eval_debug_assertions() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmy_inv::_eval_debug_assertions\n"); );
    // Body
    if (VL_UNLIKELY((in_ & 0xfeU))) {
	Verilated::overWidthError("in_");}
}
#endif // VL_DEBUG

void Vmy_inv::_ctor_var_reset() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmy_inv::_ctor_var_reset\n"); );
    // Body
    in_ = VL_RAND_RESET_I(1);
    out = VL_RAND_RESET_I(1);
}
