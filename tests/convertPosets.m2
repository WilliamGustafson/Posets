needsPackage "Python"
needsPackage "Posets"

macPosetToPython = (P) -> (
	posets = import "posets";
	posets@@Poset(toPython entries (P.RelationMatrix - transpose P.RelationMatrix) , toPython Q.GroundSet)
	);

pythonPosetToMac = (P) -> (
	poset(value P@@elements, value P@@relations(), matrix value P@@zeta())
	);
	
