# List of first names and gender

The original database can be found inside the `original_data.zip` archive or downloaded [here](ftp://ftp.heise.de/pub/ct/listings/0717-182.zip).

## Syntax

The actual database `dict.csv` is a preprocessed file to decrease the total file size and read time.
The data is semicolon separated and consists of the column `name` and `gender`.
The column gender can be one of the following values:

Value | Description
---   | --- 
M     | male first name                                                             
1M    | male name, if first part of name; else: mostly male name
?M    | mostly male name (= unisex name, which is mostly male)
F     | female first name
1F    | female name, if first part of name else: mostly female name
?F    | mostly female name (= unisex name, which is mostly female
?     | unisex name (= can be male or female)
=     | Syntax for "equivalent" names <short_name> <long_name>

# License               
List of first names and gender

Copyright (c)
2007-2008:  Jörg MICHAEL, Adalbert-Stifter-Str. 11
            30655 Hannover, 

SCCS: @(#) nam_dict.txt  1.2  2008-11-

This file is subject to the GNU Free Documentation License
Permission is granted to copy, distribute and/or 
this document under the terms of the GNU Free 
License, Version 1.2 or any later version published by 
Free Software Foundation; with no Invariant Sections
no Front-Cover Texts, and no Back-Cover Texts

This file is distributed in the hope that it will be useful
but WITHOUT ANY WARRANTY; without even the implied 
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE

A copy of the license can be found in the file GNU_DOC.TXT
You should have received a copy of the GNU Free 
License along with this file
if not, write to the  Free Software Foundation, Inc
675 Mass Ave, Cambridge, MA 02139, USA

There is one important restriction
If you modify this file in any way (e.g. add some data)
you must also release the changes under the terms of 
GNU Free Documentation License

That means you have to give out your changes, and a very 
way to do so is mailing them to the address given below
I think this is the best way to promote further 
and use of this file

If you have any remarks, feel free to e-mail to
    ct@ct.heise.

The author's email address is
   astro.joerg@googlemail.


# Famous People
The file `famous_people.txt` is obtained from [here](https://artofmemory.com/files/forum/947/initials.txt) and preprocessed.
One line contains the name of one famous person.
