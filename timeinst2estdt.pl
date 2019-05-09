#use Getopt::Std;
#use strict;
while (<>) {
    chomp;
    if (m/[0-9]:([0-9])[\s+]+([0-9.]+)\s+\((\d+)_(\d+)-(\d+)-(\d+)\)\s*$/) {
	#print "$3,$4/$5/$6,$1\n";
	if ($status{$3}==0 && $1==1) {
	    $date{$3} = "$4/$5/$6";
	}
	$status{$3} = $1; # will keep the last status
    }
}

foreach $mcn (sort keys %status) {
    print "$mcn,$date{$mcn},$status{$mcn}\n";
}
