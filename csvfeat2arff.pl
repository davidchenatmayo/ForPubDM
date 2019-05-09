####################################
# csvfeat2arff.pl
### this script uses several modules
### easiest way to install these modules:
### > curl -L http://cpanmin.us | perl - --sudo App::cpanminus
### > sudo cpanm Foo::Bar
####################################

use lib "/home/swu/perl5/lib/perl5";
use List::MoreUtils 'pairwise';
use Getopt::Std;
use Data::Dumper;
use Statistics::KernelEstimation;
use Date::Manip qw(ParseDate UnixDate);
use Date::Calc qw(Delta_Days);

my $agg="s";
my $mcnWrite=1;
my $dateWrite=0;
my $timeWrite=0;
my $pdfWrite=0;
my $bandwidth="d";
my $cdfNotPdf=1;
my $nonCausal=0;
my $statusPdfs=0;
my $useZeroValueFeatures=0;
my $inFile="srcdata/features_devset127.csv";  #"fea_sept2012_real.arff";
my $gsFile="srcdata/goldtiming_dev127.csv";  #"fea_sept2012_real.arff";
my $testFile="srcdata/features_devset127.csv"; # default to fully preprocessing the dev file to build model
my $testGsFile="srcdata/goldtiming_dev127.csv"; # default to dev
getopts("i:a:m:dtg:f:pw:n");
if ($opt_i) {
    $inFile = $opt_i;
    open INFILE, $inFile or die "Couldn't find input file!";
} else {
    open(INFILE,  "<&STDIN" )   or die "Couldn't dup STDIN : $!";
}
if ($opt_a) {
    $agg = lc(substr($opt_a,0,1));
    if ($agg eq "e") {
        if ($opt_a =~ m/cdf/) {
	    $cdfNotPdf=1;
	} else {
	    $cdfNotPdf=0;
	}
        if ($opt_a =~ m/nc/) {
	    $nonCausal=1;
	} else {
	    $nonCausal=0;
	}
        if ($opt_a =~ m/\+print/) {
	    $pdfWrite=1;
	} else {
	    $pdfWrite=0;
	}
        if ($opt_a =~ m/st/) {
	    $statusPdfs=1;
	    if ($opt_a =~ m/st2/) {
		$useZeroValueFeatures=1;
	    }
	} else {
	    $statusPdfs=0;
	}
    }
#    print stderr "Aggregation method: $agg, cdfNotPdf=$cdfNotPdf\n";
}
if ($opt_m) {
#    print stderr "ARGS: outputting mcn\n";
    $mcnWrite=$opt_m eq 0? 0:1;
#    $mcnWrite=1;
}
if ($opt_d) {
#    print stderr "ARGS: outputting date\n";
    $dateWrite=1;
}
if ($opt_t) {
#    print stderr "ARGS: printing an instance for every time point\n";
    $timeWrite=1;
}
if ($opt_g) {
    $gsFile=$opt_g;
#    print stderr "ARGS: checking $gsFile for gold standard annotations\n";
}
if ($opt_f) {
    my @files = split("[;:,]",$opt_f);
    $testFile=$files[0];
    if ($#files>0) {
	$testGsFile=$files[1];
    }
#    print stderr "ARGS: checking $testFile for test features and $testGsFile for gold\n";
} else {
    $testFile = $inFile;
    $testGsFile = $gsFile;
}
if ($opt_p) {
    $pdfWrite=1;
}
if ($opt_w) {
    $bandwidth = lc(substr($opt_w,0,1));
}
if ($opt_n) {
    $nonCausal = 1;
#    print stderr "ARGS: nonCausal!\n";
}

use strict;

##### import gold standard
my %goldstd;
my %goldtiming;
my %lastdoctiming;
open GS, $gsFile or die;
while (<GS>) {
    s/[\r\n]//g;
    chomp;
    my @parts = split(",");
    my $mcn = shift(@parts);
    my $class = pop(@parts);
    $goldstd{$mcn}=$class;

    # if there is index date info, keep it. don't worry now whether inddt and asthmabi match up
    my $inddt = shift(@parts);
    if ($inddt ne "") {
	$goldtiming{$mcn}=$inddt;
    }
}
close GS;

##### set up arff header
my $prehead = "\@RELATION AsthmaClassification"."\n".
    "\n".
    "%mcn is not a feature but will be used to track the example\n";
$prehead .= ($mcnWrite==1)? "\@ATTRIBUTE mcn string\n" : "";
my $posthead = "\@ATTRIBUTE CLASS {0,1}\n\n\@DATA\n";
my @header = (
    "\@ATTRIBUTE ASTHMA",
    "\@ATTRIBUTE COPD",
    "\@ATTRIBUTE BRONCHOSPASM",
    "\@ATTRIBUTE COUGH",
    "\@ATTRIBUTE WHEEZE",
    "\@ATTRIBUTE DYSPNEA",
    "\@ATTRIBUTE CRITERIA1",
    "\@ATTRIBUTE NIGHTTIME",
    "\@ATTRIBUTE NONSMOKER",
    "\@ATTRIBUTE NASAL_POLYPS",
    "\@ATTRIBUTE EOSINOPHILIA_HIGH",
    "\@ATTRIBUTE POSITIVE_SKIN",
    "\@ATTRIBUTE SERUM_IGE_HIGH",
    "\@ATTRIBUTE HAY_FEVER",
    "\@ATTRIBUTE INFANTILE_ECZEMA",
    "\@ATTRIBUTE EXPOSURE_TO_ANTIGEN",
    "\@ATTRIBUTE PULMONARY_TEST",
    "\@ATTRIBUTE FEV1_INCREASE",
    "\@ATTRIBUTE FVC_INCREASE",
    "\@ATTRIBUTE FEV1_DECREASE",
    "\@ATTRIBUTE PULMONARY_LOW_IMPROVEMENT",
    "\@ATTRIBUTE METHACHOLINE_TEST",
    "\@ATTRIBUTE METHACHOLINE_FEV1_LOW",
    "\@ATTRIBUTE BRONCHODILATOR",
    "\@ATTRIBUTE FAVORABLE_RESPONSE",
    "\@ATTRIBUTE BRONCHODILATOR_RESPONSE");

if ($agg ne "e" || $pdfWrite==0) {
    print $prehead;
    if ($agg eq "o") {
	foreach my $line (@header) {
	    print "$line {0,1}\n";
	}
    } elsif ($agg eq "s" || $agg eq "e") {
	foreach my $line (@header) {
	    print "$line real\n";
	}
    } elsif ($agg eq "b") {
	foreach my $line (@header) {
	    print "$line\_SUM real\n";
	}
	foreach my $line (@header) {
	    print "$line\_OR {0,1}\n";
	}
    }
    print $posthead;
}

####################################
##### read and print the data
####################################
my $NUMFEATS = @header;
my %ptfeats;
my %helper1;
my %helper2; 
my @posStatusPosFeatPdfs;
my @negStatusPosFeatPdfs;
my @posStatusNegFeatPdfs;
my @negStatusNegFeatPdfs;
my @posFeatCtr; # currently: count all cases, both pos and neg
my @bws; # bandwidths
my %posDatesInFile;
my %negDatesInFile;
my %allDatesInFile;

while(<INFILE>) {
    chomp;
    (my $mcn, my $date, my $docid, my @feats) = split(",");
#    ($#feats == 26) or die;
    $mcn =~ s/^0*//g;

    if ($agg eq "s") {
	my @sumfeats = pairwise { $a + $b } @{$ptfeats{$mcn}}, @feats;
	$ptfeats{$mcn} = [@sumfeats];
    } elsif ($agg eq "o") {
	my @orfeats =  pairwise { $a || $b } @{$ptfeats{$mcn}}, @feats;
	$ptfeats{$mcn} = [@orfeats];
    } elsif ($agg eq "b") {
	my @sumfeats = pairwise { $a + $b } @{$helper1{$mcn}}, @feats;
	my @orfeats =  pairwise { $a || $b } @{$helper2{$mcn}}, @feats;
	$helper1{$mcn} = [@sumfeats];
	$helper2{$mcn} = [@orfeats];
	$ptfeats{$mcn} = [@sumfeats, @orfeats];
    } elsif ($agg eq "e") {

	$allDatesInFile{$mcn} .= ($allDatesInFile{$mcn} eq ""? "":",").$date;


#	# unlike all the others, estpdf will store the data and create models
#	if ($date ne "" && $goldtiming{$mcn} ne "") { # ensure that datediff is valid

	    for (my $i=0; $i<=$#feats; $i++) {

		# store all the features up to this point
		if ($feats[$i] ne "0") {
		    $posDatesInFile{$mcn}[$i] .= ($posDatesInFile{$mcn}[$i] eq ""? "":",").$date;
		    $negDatesInFile{$mcn}[$i] .= "";
		} else {
		    $posDatesInFile{$mcn}[$i] .= "";
		    $negDatesInFile{$mcn}[$i] .= ($negDatesInFile{$mcn}[$i] eq ""? "":",").$date;
		}
	    }

    }

    # print out msp (mult. statuses per pt.) version for non-pdf versions
    if ($timeWrite==1 && $agg ne "e") {
	printone($mcn,$date);
    }
}

##### print out ssp (single status per pt.) version
if ($timeWrite==0 && $agg ne "e") {
    printagg();
}


##### estimate the status pdfs
if ($agg eq "e") { 
    foreach my $mcn (keys %allDatesInFile) {
	my $dates = $allDatesInFile{$mcn};
	for my $inferenceDate (split(",", $dates)) {
	    
	    for (my $i=0; $i<$NUMFEATS; $i++) {
		
		if (!$posStatusPosFeatPdfs[$i]) {
		    $posStatusPosFeatPdfs[$i] = Statistics::KernelEstimation->new();
		    $negStatusPosFeatPdfs[$i] = Statistics::KernelEstimation->new();
		    $posStatusNegFeatPdfs[$i] = Statistics::KernelEstimation->new();
		    $negStatusNegFeatPdfs[$i] = Statistics::KernelEstimation->new();
		}

		# pdfs from positive feature status: P( s(t)=1 | f(0)=1 ) and P( s(t)=0 | f(0)=1 )
		my $posFeatDates = $posDatesInFile{$mcn}[$i];
		if ($posFeatDates) {
		    for my $posFeatDate (split(",",$posFeatDates)) {

			# positive case: s(t)==1 AND f(0)==1
			if ($goldtiming{$mcn} ne "" && datediff($goldtiming{$mcn},$inferenceDate)>0) {

			    $posStatusPosFeatPdfs[$i]->add_data( datediff($posFeatDate,$inferenceDate) );
			    $posFeatCtr[$i]++;

#			    print STDERR "$inferenceDate was a positive example, so comparing with $posFeatDate -> $dd\n";
			}
			# negative case: s(t)==0 BUT f(0)==1
			else {
			    
			    $negStatusPosFeatPdfs[$i]->add_data( datediff($posFeatDate,$inferenceDate) );
			    $posFeatCtr[$i]++;

			}
			
		    } 

		}
		# pdfs from negative feature status: P( s(t)=1 | f(0)=0 ) and P( s(t)=0 | f(0)=0 )
		my $negFeatDates = $negDatesInFile{$mcn}[$i];
		if ($negFeatDates) {
		    for my $negFeatDate (split(",",$negFeatDates)) {

			# s(t)==1 BUT f(0)==0
			if ($goldtiming{$mcn} ne "" && datediff($goldtiming{$mcn},$inferenceDate)>0) {
			    $posStatusNegFeatPdfs[$i]->add_data( datediff($negFeatDate,$inferenceDate) );
			}
			# s(t)==0 AND f(0)==0
			else {
			    $negStatusNegFeatPdfs[$i]->add_data( datediff($negFeatDate,$inferenceDate) );
			}
			
		    } 

		}
	    }
	}
    }
}





####################################
##### re-analyze based on the estimated pdfs
####################################
if ($agg eq "e" && $pdfWrite==1) {
    for (my $i=0; $i<$NUMFEATS; $i++) {

	# default case
	if ($statusPdfs==0) {
#	    if ($posStatusPosFeatPdfs[$i]->count > 10) {
	    if ($posStatusPosFeatPdfs[$i]->count > 1) {
		if ($bandwidth eq "o") {
		    $bws[$i] = $posStatusPosFeatPdfs[$i]->optimal_bandwidth();
		} else {
		    $bws[$i] = $posStatusPosFeatPdfs[$i]->default_bandwidth();
		}
		if ($pdfWrite==1) {
		    print STDERR "writing pdfs to file\n";
		    printpdf($posStatusPosFeatPdfs[$i],$bws[$i],$i);
		}
	    } else {
		$bws[$i] = 0;
	    }
	} 
	# negative pdfs should be written
	else {
	    if ($negStatusPosFeatPdfs[$i]->count > 10) {
		if ($bandwidth eq "o") {
		    $bws[$i] = $negStatusPosFeatPdfs[$i]->optimal_bandwidth();
		} else {
		    $bws[$i] = $negStatusPosFeatPdfs[$i]->default_bandwidth();
		}
		if ($pdfWrite==1) {
		    print STDERR "writing negStatusPosFeatPdfs to file\n";
		    printscaledpdf($posStatusPosFeatPdfs[$i],$negStatusPosFeatPdfs[$i],$bws[$i],$i);
		}
	    } else {
		$bws[$i] = 0;
	    }
	}
    }
    if ($pdfWrite==1) {exit;}

} elsif ($agg eq "e") {
    print STDERR "INFO: pdfs are estimated, now need to generate weighted features from test file $testFile.\n";

    my %featDates;
    my %negDates;
    my %allDates;

    # if we're running on a separate testset, then we will need another gold standard file
    if ($testGsFile ne $gsFile) {
	open TESTGS, $testGsFile or die;
	while (<TESTGS>) {
	    s/[\r\n]//g;
	    chomp;
	    
	    my @parts = split(",");
	    my $mcn = shift(@parts);
	    my $class = pop(@parts);
	    $goldstd{$mcn}=$class;
	    
	    # if there is index date info, keep it. don't worry now whether inddt and asthmabi match up
	    my $inddt = shift(@parts);
	    if ($inddt ne "") {
		$goldtiming{$mcn}=$inddt;
	    }
# print "stored $mcn $inddt $class\n";
	}
	close TESTGS;
    }

    # need to look at a test set of data; but will run on the same data by default
    open FTF, $testFile or die "ERROR: estimated pdfs but had no features testFile to run on. que lastima!";
    while (<FTF>) {
	
	chomp;
	(my $mcn, my $date, my $docid, my @feats) = split(",");
	$mcn =~ s/^0*//g;
	$lastdoctiming{$mcn} = $date;

	if ($nonCausal==1) {
	    $allDates{$mcn} .= ($allDates{$mcn} eq ""? "":",").$date;
	}

	for (my $i=0; $i<=$#feats; $i++) {

	    # store all the features up to this point
	    if ($feats[$i] ne "0") {
		$featDates{$mcn}[$i] .= ($featDates{$mcn}[$i] eq ""? "":",").$date;
		$negDates{$mcn}[$i] .= "";
	    } else {
		$negDates{$mcn}[$i] .= ($negDates{$mcn}[$i] eq ""? "":",").$date;
		$featDates{$mcn}[$i] .= "";
	    }

	    # calculate & print if there needs to be something written out at every time
	    if ($timeWrite==1 && $nonCausal==0) {
#	    if ($timeWrite==1) {

		# recalculate this time and feature from scratch
		$ptfeats{$mcn}[$i] = 0;

		my $fdates = $featDates{$mcn}[$i];
		if ($fdates) { 
		    if ($posStatusPosFeatPdfs[$i]->count > 10) {
#		    if ($posStatusPosFeatPdfs[$i]->count > 1) {
			for my $d (split(",", $fdates)) {
			    # calculate the contribution of each old feature to this feature 
			    my $datediff = datediff($d,$lastdoctiming{$mcn});
#			    print stderr "datediff $datediff. should skip date $d if it's after $lastdoctiming{$mcn}\n";
			    if ($datediff<0) {
				next;
			    }
			    if ($cdfNotPdf==1) {
				$ptfeats{$mcn}[$i] += $posStatusPosFeatPdfs[$i]->cdf($datediff, $bws[$i]);
			    } elsif ($statusPdfs==1) {
				# "negative" version scales the pdf by the likelihood at this timepoint that the feat==1 vs feat==0
				$ptfeats{$mcn}[$i] += $posStatusPosFeatPdfs[$i]->pdf($datediff, $bws[$i]) /
				    ($posStatusPosFeatPdfs[$i]->pdf($datediff,$bws[$i]) + $negStatusPosFeatPdfs[$i]->pdf($datediff,$bws[$i]));
			    } else {
				$ptfeats{$mcn}[$i] += $posStatusPosFeatPdfs[$i]->pdf($datediff, $bws[$i]);
			    }
			}
		    } else {
			# default to summative model if no reliable pdf
			$ptfeats{$mcn}[$i]++;
		    }
		} else {
		    $ptfeats{$mcn}[$i] = 0;
		}


	    }
	}
	if ($timeWrite==1 && $nonCausal==0) {
#	if ($timeWrite==1) {
	    printone($mcn,$date);
	}

    }

    # no calculations have been made yet, dates and pdf are prepped... now, calc and write!
    if ($timeWrite==0) {
	foreach my $mcn (keys %featDates) {
	    for (my $i=0; $i<$NUMFEATS; $i++) {
		my $fdates = $featDates{$mcn}[$i];
		if ($fdates) { 
		    for my $d (split(",", $fdates)) {
			# calculate the contribution of each old feature to this feature 
			if ($posStatusPosFeatPdfs[$i]->count > 10) {
			    my $datediff = datediff($d,$lastdoctiming{$mcn});
			    if ($cdfNotPdf==1) {
				$ptfeats{$mcn}[$i] += $posStatusPosFeatPdfs[$i]->cdf($datediff, $bws[$i]);
			    } elsif ($statusPdfs==1) {
				# "negative" version scales the pdf by the likelihood at this timepoint that the feat==1
				$ptfeats{$mcn}[$i] += $posStatusPosFeatPdfs[$i]->pdf($datediff, $bws[$i]) /
				    ($posStatusPosFeatPdfs[$i]->pdf($datediff,$bws[$i]) + $negStatusPosFeatPdfs[$i]->pdf($datediff,$bws[$i]));
			    } else {
				$ptfeats{$mcn}[$i] += $posStatusPosFeatPdfs[$i]->pdf($datediff, $bws[$i]);
			    }
			} else {
			    # default to additive model if no reliable pdf
			    $ptfeats{$mcn}[$i]++;
			}

		    }
		} else {
		    $ptfeats{$mcn}[$i] = 0;
		}

	    }
	}
	printagg();

    } 
    ### non causal version
    elsif ($timeWrite==1 && $nonCausal==1) {
	foreach my $mcn (keys %allDates) {
	    my $dates = $allDates{$mcn};
	    for my $cd (split(",", $dates)) {
#		my $i=0;
		for (my $i=0; $i<$NUMFEATS; $i++) {
		
		    $ptfeats{$mcn}[$i] = 0;
		    my $fds = $featDates{$mcn}[$i];
		    if ($fds) {
			for my $fd (split(",",$fds)) {
			    
#			    print "mcn $mcn, date $cd, feat $i, featDate $fd\n";
		    
			    if ($posStatusPosFeatPdfs[$i]->count > 10) {
#			    if ($posStatusPosFeatPdfs[$i]->count > 1) {
				my $datediff = datediff($fd,$cd);
				$ptfeats{$mcn}[$i] += $posStatusPosFeatPdfs[$i]->pdf($datediff, $bws[$i]);
				
			    } else {
				# default to additive model
				$ptfeats{$mcn}[$i]++;
			    }
			
			}
			
		    } else {
			#for (my $i=0; $i<$NUMFEATS; $i++) { $ptfeats{$mcn}[$i] = 0; }
			$ptfeats{$mcn}[$i] = 0;
		    }
		}

		printone($mcn,$cd);

	    }
	}
    }

}

####################################
##### subroutines
####################################


sub datediff {
    my $date1 = ParseDate($_[0]);
    my $date2 = ParseDate($_[1]);
    if (!$date1 || !$date2) {
	warn "Bad date string: $_[0] or $_[1]\n";
	next;
    } else {
	return Delta_Days(UnixDate($date1,"%Y","%m","%d"),UnixDate($date2,"%Y","%m","%d"));
    }
}

sub printpdf {
    my $pdf = $_[0];
    my $w = $_[1];
    my $prepend = $_[2];
    if (!$w) { $w = $pdf->default_bandwidth(); }
    ( my $min, my $max ) = $pdf->extended_range();
    
    if ($cdfNotPdf==1) {
	for( my $x=$min; $x<=$max; $x+=($max-$min)/100 ) {
	    print $prepend.($prepend ne ""?",":""), $x, ",", $pdf->cdf( $x, $w ), "\n";
	}
    } else {
	for( my $x=$min; $x<=$max; $x+=($max-$min)/100 ) {
	    print $prepend.($prepend ne ""?",":""), $x, ",", $pdf->pdf( $x, $w ), "\n";
	}
    }
}

sub printscaledpdf {
    my $pdf1 = $_[0];
    my $pdf2 = $_[1];
    my $w = $_[2];
    my $prepend = $_[3];
    if (!$w) { $w = $pdf1->default_bandwidth(); }
    ( my $min, my $max ) = $pdf1->extended_range();
    
    if ($cdfNotPdf==1) {
	for( my $x=$min; $x<=$max; $x+=($max-$min)/100 ) {
	    print $prepend.($prepend ne ""?",":""), $x, ",", $pdf1->cdf( $x, $w ), "\n";
	}
    } elsif ($statusPdfs==1) {
	for( my $x=$min; $x<=$max; $x+=($max-$min)/100 ) {
	    print $prepend.($prepend ne ""?",":""), $x, ",", 
	    $pdf1->pdf($x,$w) / ($pdf1->pdf($x,$w) + $pdf2->pdf($x,$w)), 
	    "\n";
	}
    } else {
	for( my $x=$min; $x<=$max; $x+=($max-$min)/100 ) {
	    print $prepend.($prepend ne ""?",":""), $x, ",", $pdf1->pdf( $x, $w ), "\n";
	}
    }
}

sub printagg {
    my $date = $_[0];
    for my $mcn (sort keys %ptfeats) {
	if ($dateWrite==1 && $timeWrite==0) {
	    print STDERR "WARNING: cannot write date on fully aggregated patient data\n";
	}
	if ($mcnWrite==1) {
	    print $mcn.",";
	}
	print join(",",@{$ptfeats{$mcn}},$goldstd{$mcn})."\n";
    }
}

sub printone {
    my ($mcn, $date) = @_;
    my $class;
    # mark the correct timing for the gold standard
    if ($goldstd{$mcn}==1 && datediff($goldtiming{$mcn},$date)<0) {
	$class = 0;
    } else {
	$class = $goldstd{$mcn};
    }
    print ($mcnWrite==1?$mcn:"");
    print (($dateWrite==1 && $timeWrite==1)?"_$date":"");
    print (($mcnWrite==1 || $dateWrite==1)?",":"");
    print join(",",@{$ptfeats{$mcn}},$class)."\n";
}
