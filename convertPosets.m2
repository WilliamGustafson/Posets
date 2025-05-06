needsPackage "Python"
needsPackage "Posets"

macPosetToPython = (P) -> (
	posets = import "posets";
	M=entries P.RelationMatrix;
	zetamat={};
	for i from 0 to #M-1 do zetamat = append(zetamat,M_i_{i..#M-1});
	return posets@@Poset(toPython zetamat, toPython P.GroundSet);
	);

pythonPosetToMac = (P) -> (
	return poset(value P@@elements, value P@@relations())
	);
	
