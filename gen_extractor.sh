genotypes=$(cat pre_trained_model/search-EXP-20210124-131121/log.txt | grep "Genotype(" | sed "s/^.*\(genotype =.*\)$/\1/g")
cpt=0
while IFS= read -r line
do
    echo $line | sed "s/^genotype/GEN_$cpt/"
    cpt=$((cpt + 1))
done <<< $"$genotypes"
