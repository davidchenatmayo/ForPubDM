while(<>) {
    chomp;
    if (m/\d+\s+([0-9:]+)\s+([0-9:]+)\s+\+?\s+[0-9.]+\s+\(([0-9_-]+)\)/) {
#	print $1, " ", $3, "\n";
	($mcn, $date) = split("_",$3);
	if ($1 eq "2:1" && $2 eq "2:1") {
	    $tp{$mcn}++;
	} elsif ($1 eq "2:1" && $2 eq "1:0") {
	    $fn{$mcn}++;
	} elsif ($1 eq "1:0" && $2 eq "2:1") {
	    $fp{$mcn}++;
	} elsif ($1 eq "1:0" && $2 eq "1:0") {
	    $tn{$mcn}++;
	}
	$total{$mcn}++;
    }
}

foreach $mcn (keys %total) {
#    $ppv{$mcn} = $tp{$mcn} / ($tp{$mcn}+$fp{$mcn});
#    $sens{$mcn} = $tp{$mcn} / ($tp{$mcn}+$fn{$mcn});
# 
#    $spec{$mcn} = $tn{$mcn} / ($tn{$mcn}+$fp{$mcn});
#    $npv{$mcn} = $tn{$mcn} / ($tn{$mcn}+$fn{$mcn});

    $acc{$mcn} = ($tp{$mcn}+$tn{$mcn}) / $total{$mcn};

    # calculate "balanced" tp/fn/fp/tn
    $tp{$mcn} /= $total{$mcn};
    $fn{$mcn} /= $total{$mcn};
    $fp{$mcn} /= $total{$mcn};
    $tn{$mcn} /= $total{$mcn};

#    print "$tp{$mcn},$fp{$mcn},$fn{$mcn},$tn{$mcn},$total{$mcn} acc=$acc{$mcn}\n";

#    $ppvsum += $ppv{$mcn};
#    $senssum += $sens{$mcn};
#    $specsum += $spec{$mcn};
#    $npvsum += $npv{$mcn};
    $accsum += $acc{$mcn};

    $tpsum += $tp{$mcn};
    $fnsum += $fn{$mcn};
    $fpsum += $fp{$mcn};
    $tnsum += $tn{$mcn};
}

#$ppvsum /= (keys %total);
#$senssum /= (keys %total);
#$specsum /= (keys %total);
#$npvsum /= (keys %total);
#print "$senssum,$specsum,$ppvsum,$pnvsum,",(2*($ppvsum+$senssum)/($ppvsum*$senssum)),"\n";

$sens  = $tpsum / ($tpsum+$fnsum) *100;
$spec = $tnsum / ($tnsum+$fpsum) *100;
$ppv = $tpsum / ($tpsum+$fpsum) *100;
$npv  = $tnsum / ($tnsum+$fnsum) *100;
$f1 = (2*($ppv*$sens)/($ppv+$sens));
$acc = $accsum/(keys %total) *100;

printf("%.2f%%,%.2f%%,%.2f%%,%.2f%%,%.2f%%,%.2f%%\n",$sens,$spec,$ppv,$npv,$f1,$acc);
