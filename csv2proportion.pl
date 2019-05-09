
while(<>) {
    chomp;
    ($mcn,$date,$docno,@feats) = split ",";
    $numsamples++;
#    print join(" ",$numsamples,$mcn,$date,$docno,"|||",join(" ",@feats))."\n";
    for ($i=0;$i<@feats;$i++) {
	if ($feats[$i] eq "1") {
	    $possamples[$i]++;
	}
    }
}
for ($i=0;$i<@feats;$i++) {
    print join(",",$i,$possamples[$i],$numsamples,$possamples[$i]/$numsamples)."\n";
}
