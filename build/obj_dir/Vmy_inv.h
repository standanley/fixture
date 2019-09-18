// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Primary design header
//
// This header should be included by all source files instantiating the design.
// The class here is then constructed to instantiate the design.
// See the Verilator manual for examples.

#ifndef _Vmy_inv_H_
#define _Vmy_inv_H_

#include "verilated.h"

class Vmy_inv__Syms;

//----------

VL_MODULE(Vmy_inv) {
  public:
    
    // PORTS
    // The application code writes and reads these signals to
    // propagate new values into/out from the Verilated model.
    VL_IN8(in_,0,0);
    VL_OUT8(out,0,0);
    
    // LOCAL SIGNALS
    // Internals; generally not touched by application code
    
    // LOCAL VARIABLES
    // Internals; generally not touched by application code
    
    // INTERNAL VARIABLES
    // Internals; generally not touched by application code
    Vmy_inv__Syms* __VlSymsp;  // Symbol table
    
    // PARAMETERS
    // Parameters marked /*verilator public*/ for use by application code
    
    // CONSTRUCTORS
  private:
    Vmy_inv& operator= (const Vmy_inv&);  ///< Copying not allowed
    Vmy_inv(const Vmy_inv&);  ///< Copying not allowed
  public:
    /// Construct the model; called by application code
    /// The special name  may be used to make a wrapper with a
    /// single model invisible WRT DPI scope names.
    Vmy_inv(const char* name="TOP");
    /// Destroy the model; called (often implicitly) by application code
    ~Vmy_inv();
    
    // API METHODS
    /// Evaluate the model.  Application must call when inputs change.
    void eval();
    /// Simulation complete, run final blocks.  Application must call on completion.
    void final();
    
    // INTERNAL METHODS
  private:
    static void _eval_initial_loop(Vmy_inv__Syms* __restrict vlSymsp);
  public:
    void __Vconfigure(Vmy_inv__Syms* symsp, bool first);
  private:
    static QData _change_request(Vmy_inv__Syms* __restrict vlSymsp);
  public:
    static void _combo__TOP__1(Vmy_inv__Syms* __restrict vlSymsp);
  private:
    void _ctor_var_reset();
  public:
    static void _eval(Vmy_inv__Syms* __restrict vlSymsp);
  private:
#ifdef VL_DEBUG
    void _eval_debug_assertions();
#endif // VL_DEBUG
  public:
    static void _eval_initial(Vmy_inv__Syms* __restrict vlSymsp);
    static void _eval_settle(Vmy_inv__Syms* __restrict vlSymsp);
} VL_ATTR_ALIGNED(128);

#endif // guard
