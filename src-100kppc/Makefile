MPIF90 ?= mpif90
MPICC ?= mpicc
# OPTS = -O2 
# COPTS = -O2 
OPTS = -O2 -march=core-avx2 -mtune=core-avx2 -traceback# -g -C
COPTS = -O2 -march=core-avx2 -mtune=core-avx2 -traceback # -g -C
#OPTS = -O0 -no-openmp -march=core-avx2 -mtune=core-avx2 -traceback -debug full -g -C
#COPTS = -O0 -no-openmp -march=core-avx2 -mtune=core-avx2 -traceback -debug full -g -C

MODS := beaminplasma.mod currentproblemvalues.mod diagnostics.mod electroninjection.mod ioninducedseemission.mod langevincollisions.mod maxwellenergy.mod maxwellvelocity.mod mccollisions.mod paralleloperationvalues.mod seemission.mod snapshots.mod testparticles.mod rng_wrapper.mod

SRC := parRandomNumberInterface.f90 parBeamInjection.f90 parBeamInPlasma.f90 parCheckpoint.f90 parCollisionProc.f90 parCurProbVals.f90 parDiagNoGraph.f90 parFieldCalcProc.f90 parIonInducedSEEmission.f90 parLangevinColl.f90 parMainProgram.f90 parMCCollisions.f90 parModules.f90 parPartDynamicProc.f90 parPartExchangeProc.f90 parPlasmaSourceProc.f90 parSEEmission.f90 parSnapshot.f90

OBJS = $(SRC:.f90=.o) WELL19937a_new.o

default : edipic

$(MODS) : parModules.f90
	$(MPIF90) $(OPTS) -c $<

WELL19937a_new.o : WELL19937a_new.c
	$(MPICC) $(COPTS) -c $<

edipic : $(SRC) $(MODS) WELL19937a_new.o
	$(MPIF90) $(OPTS) -o edipic $(SRC) WELL19937a_new.o

clean :
	-rm -f $(MODS) $(OBJS) edipic
