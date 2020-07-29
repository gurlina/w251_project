#!/bin/bash

while getopts f:t: flag
do
     case "${flag}" in
         f) fromdir=${OPTARG};;
         t) todir=${OPTARG};;
     esac
done

[[ "${fromdir}" != */ ]] && fromdir="${fromdir}/"
todir=${todir%/}

mkdir -p $todir/fake
mkdir -p $todir/real

startcolor="\e[30;48;5;45m"
endcolor="\e[0m"

echo -e "Copying $startcolor real images $endcolor from $startcolor $fromdir $endcolor to $startcolor $todir/real $endcolor ..."

find $fromdir -type f -name '*real*.png' -print0 | xargs -0 -I{} cp {} $todir/real

echo -e "Finished copying real images."

count=$(ls $todir/real | wc -l)
echo -e "There are $startcolor $count $endcolor of real images."

echo -e "Copying $startcolor fake images $endcolor from $startcolor $fromdir $endcolor to $startcolor $todir/fake $endcolor ..."

find $fromdir -type f -name '*fake*.png' -print0 | xargs -0 -I{} cp {} $todir/fake

echo -e "Finished copying fake images."

count=$(ls $todir/fake | wc -l)
echo -e "There are $startcolor $count $endcolor of fake images."
