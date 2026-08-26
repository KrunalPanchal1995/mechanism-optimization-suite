# W7c fixture: an RMG input file with the P39 pathologies —
# 'GRI-Mech3.0' dual-listed in reactionLibraries AND seedMechanisms
# (RMG-Py #2979/#2882), a duplicate thermo library (#2485), and inverted
# interrupt/move-to-core tolerances.

database(
    thermoLibraries=['primaryThermoLibrary', 'BurkeH2O2', 'BurkeH2O2'],
    reactionLibraries=[('GRI-Mech3.0', False), 'BurkeH2O2inN2'],
    seedMechanisms=['GRI-Mech3.0'],
    kineticsDepositories=['training'],
    kineticsFamilies='default',
    kineticsEstimator='rate rules',
)

species(
    label='ethane',
    reactive=True,
    structure=SMILES("CC"),
)

simpleReactor(
    temperature=(1350, 'K'),
    pressure=(1.0, 'bar'),
    initialMoleFractions={"ethane": 1.0},
    terminationConversion={'ethane': 0.9},
    terminationTime=(1e6, 's'),
)

model(
    toleranceKeepInEdge=0.0,
    toleranceMoveToCore=0.5,
    toleranceInterruptSimulation=0.1,
    maximumEdgeSpecies=100000,
)
