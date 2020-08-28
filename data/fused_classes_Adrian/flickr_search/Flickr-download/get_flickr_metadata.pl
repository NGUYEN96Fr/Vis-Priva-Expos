#!/usr/bin/perl
use POSIX;
use URI::Escape;
# script for getting Flickr metadata for the persons in the list that is used for learning

$id_list = @ARGV[0]; # list of personalities
$odir = @ARGV[1]; #output directory that will contain metadata for the current user
$max_pages = @ARGV[2]; #maximum number of pages that will be downloaded for each user
$list_batch = @ARGV[3]; #number of the batch downloaded with the current instance of the script
$total_batches = @ARGV[4]; # total number of batches

# USAGE: 
#perl get_flickr_metadata.pl files/flickr_slist.txt /home/vankhoa/Vis-Priva-Expos/data/Files_BERGAMOTE/Flickr_download/flickr_metadata 6 0 1

# select which API key should be used
if($list_batch == 0){$api_key = "78375e7f32b84f9f8ae7b497bc1af167";}
elsif($list_batch == 1){$api_key = "4a697f5c92cc1c7489202e270eed781e";}
elsif($list_batch == 2){$api_key = "2f34564106fcb03667ecb0aaf57f9f74";}
elsif($list_batch == 3){$api_key = "8640b9adbd5f2cfcd0235d3c3853f146";}
elsif($list_batch == 4){$api_key = "78375e7f32b84f9f8ae7b497bc1af167";}
elsif($list_batch == 5){$api_key = "4a697f5c92cc1c7489202e270eed781e";}

#create the output directory
if(!(-e $odir)){mkdir $odir;}

#open the list of ids
open UL, $id_list or die "cant open user list $id_list\n";
@ids = <UL>;
close UL;

#We use 4 batches and split the list accordingly
$step = int(($#ids+1)/$total_batches) + 1;
$min_pos = $list_batch * $step;
$max_pos = ($list_batch+1) * $step;

if($max_pos > $#ids){$max_pos = $#ids+1;}

print "min and max pos: $min_pos, $max_pos\n";

for($pos = $min_pos; $pos < $max_pos; $pos++)
{
 $crt_name = @ids[$pos];
 chomp($crt_name);
 @name_parts = split(/\t/, $crt_name);
 $norm_name = uri_escape(@name_parts[0]);
 $norm_name =~ s/'/%27/g;
 print "$pos - @name_parts[0]; @name_parts[1]\n";
 # create a directory for the current identity
 $id_dir = "$odir/@name_parts[1]";
 if(!(-e $id_dir)){mkdir $id_dir;}
 for($pg = 1; $pg <= $max_pages; $pg++)
 {
  $query = 'https://api.flickr.com/services/rest/?method=flickr.photos.search&api_key='.$api_key.'&text=%22'.$norm_name.'%22&sort=relevance&extras=date_taken%2Ctags%2Cdate_upload%2Cviews&per_page=500&page='.$pg;
$out_file = "$id_dir/$pg";
  if(!(-e $out_file))
  {
   $toExec = "wget -q -t 1 --no-check-certificate --user-agent=\'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71 Safari/537.36\' -O $out_file \'$query\'";
   #print "$toExec\n";
   `$toExec`;
  }
 }

}
