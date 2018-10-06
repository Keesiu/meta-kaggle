# this was produced by calling kaggle_to_collected(2011) and then manually fixing it by looking at
#   http://en.wikipedia.org/wiki/List_of_colloquial_names_for_universities_and_colleges_in_the_United_States
# For schools for which I didn't find the matching the value is None

kaggle_to_collected = {
    'Abilene Chr': 'Abilene Christian Wildcats',
    'Air Force': 'Air Force Falcons',
    'Akron': 'Akron Zips',
    'Alabama': 'Alabama Crimson Tide',
    'Alabama A&M': 'Alabama A&M Bulldogs',
    'Alabama St': 'Alabama State Hornets',
    'Albany NY': 'Albany (NY) Great Danes',
    'Alcorn St': 'Alcorn State Braves',
    'Alliant Intl': None, #u'Alliant International Gulls',
    'American Univ': 'American Eagles',
    'Appalachian St': 'Appalachian State Mountaineers',
    'Arizona': 'Arizona Wildcats',
    'Arizona St': 'Arizona State Sun Devils',
    'Ark Little Rock': 'Arkansas-Little Rock Trojans',
    'Ark Pine Bluff': 'Arkansas-Pine Bluff Golden Lions',
    'Arkansas': 'Arkansas Razorbacks',
    'Arkansas St': 'Arkansas State Red Wolves',
    'UT Arlington': 'Texas-Arlington Mavericks',
    'Armstrong St': None, #u'Armstrong Pirates',
    'Army': 'Army Black Knights',
    'Auburn': 'Auburn Tigers',
    'Augusta': None, #u'Augusta State Jaguars',
    'Austin Peay': 'Austin Peay Governors',
    'BYU': 'Brigham Young Cougars',
    'Ball St': 'Ball State Cardinals',
    'Baylor': 'Baylor Bears',
    'Belmont': 'Belmont Bruins',
    'Bethune-Cookman': 'Bethune-Cookman Wildcats',
    'Binghamton': 'Binghamton Bearcats',
    'Birmingham So': None,
    'Boise St': 'Boise State Broncos',
    'Boston College': 'Boston College Eagles',
    'Boston Univ': 'Boston University Terriers',
    'Bowling Green': 'Bowling Green State Falcons',
    'Bradley': 'Bradley Braves',
    'Brooklyn': None, #u'Brooklyn Bulldogs',
    'Brown': 'Brown Bears',
    'Bryant': 'Bryant Bulldogs',
    'Bucknell': 'Bucknell Bison',
    'Buffalo': 'Buffalo Bulls',
    'Butler': 'Butler Bulldogs',
    'C Michigan': 'Central Michigan Chippewas',
    'CS Bakersfield': 'Cal State Bakersfield Roadrunners',
    'CS Fullerton': 'Cal State Fullerton Titans',
    'CS Northridge': 'Cal State Northridge Matadors',
    'CS Sacramento': 'Sacramento State Hornets',
    'Cal Poly SLO': 'Cal Poly Mustangs',
    'California': 'University of California Golden Bears',
    'Campbell': 'Campbell Fighting Camels',
    'Canisius': 'Canisius Golden Griffins',
    'Cent Arkansas': 'Central Arkansas Bears',
    'Centenary': 'Centenary (LA) Gents',
    'Central Conn': 'Central Connecticut State Blue Devils',
    'Charleston So': 'Charleston Southern Buccaneers',
    'Charlotte': 'Charlotte 49ers',
    'Chattanooga': 'Chattanooga Mocs',
    'Chicago St': 'Chicago State Cougars',
    'Cincinnati': 'Cincinnati Bearcats',
    'Citadel': 'Citadel Bulldogs',
    'Clemson': 'Clemson Tigers',
    'Cleveland St': 'Cleveland State Vikings',
    'Coastal Car': 'Coastal Carolina Chanticleers',
    'Col Charleston': 'College of Charleston Cougars',
    'Colgate': 'Colgate Raiders',
    'Colorado': 'Colorado Buffaloes',
    'Colorado St': 'Colorado State Rams',
    'Columbia': 'Columbia Lions',
    'Connecticut': 'Connecticut Huskies',
    'Coppin St': 'Coppin State Eagles',
    'Cornell': 'Cornell Big Red',
    'Creighton': 'Creighton Bluejays',
    'Dartmouth': 'Dartmouth Big Green',
    'Davidson': 'Davidson Wildcats',
    'Dayton': 'Dayton Flyers',
    'DePaul': 'DePaul Blue Demons',
    'Delaware': "Delaware Fightin' Blue Hens",
    'Delaware St': 'Delaware State Hornets',
    'Denver': 'Denver Pioneers',
    'Detroit': 'Detroit Mercy Titans',
    'Drake': 'Drake Bulldogs',
    'Drexel': 'Drexel Dragons',
    'Duke': 'Duke Blue Devils',
    'Duquesne': 'Duquesne Dukes',
    'E Illinois': 'Eastern Illinois Panthers',
    'E Kentucky': 'Eastern Kentucky Colonels',
    'E Michigan': 'Eastern Michigan Eagles',
    'E Washington': 'Eastern Washington Eagles',
    'ETSU': 'East Tennessee State Buccaneers',
    'East Carolina': 'East Carolina Pirates',
    'Edwardsville': 'Southern Illinois-Edwardsville Cougars',
    'Elon': 'Elon Phoenix',
    'Evansville': 'Evansville Purple Aces',
    'F Dickinson': 'Fairleigh Dickinson Knights',
    'FL Atlantic': 'Florida Atlantic Owls',
    'FL Gulf Coast': 'Florida Gulf Coast Eagles',
    'Fairfield': 'Fairfield Stags',
    'Florida': 'Florida Gators',
    'Florida A&M': 'Florida A&M Rattlers',
    'Florida Intl': 'Florida International Panthers',
    'Florida St': 'Florida State Seminoles',
    'Fordham': 'Fordham Rams',
    'Fresno St': 'Fresno State Bulldogs',
    'Furman': 'Furman Paladins',
    'G Washington': 'George Washington Colonials',
    'Ga Southern': 'Georgia Southern Eagles',
    'Gardner Webb': "Gardner-Webb Runnin' Bulldogs",
    'George Mason': 'George Mason Patriots',
    'Georgetown': 'Georgetown Hoyas',
    'Georgia': 'Georgia Bulldogs',
    'Georgia St': 'Georgia State Panthers',
    'Georgia Tech': 'Georgia Tech Yellow Jackets',
    'Gonzaga': 'Gonzaga Bulldogs',
    'Grambling': 'Grambling Tigers',
    'Grand Canyon': 'Grand Canyon Antelopes',
    'WI Green Bay': 'Green Bay Phoenix',
    'Hampton': 'Hampton Pirates',
    'Hardin-Simmons': None,
    'Hartford': 'Hartford Hawks',
    'Harvard': 'Harvard Crimson',
    'Hawaii': 'Hawaii Warriors',
    'High Point': 'High Point Panthers',
    'Hofstra': 'Hofstra Pride',
    'Holy Cross': 'Holy Cross Crusaders',
    'Houston': 'Houston Cougars',
    'Houston Bap': 'Houston Baptist Huskies',
    'Howard': 'Howard Bison',
    'IL Chicago': 'Illinois-Chicago Flames',
    'IPFW': 'IPFW Mastodons',
    'IUPUI': 'IUPUI Jaguars',
    'Idaho': 'Idaho Vandals',
    'Idaho St': 'Idaho State Bengals',
    'Illinois': 'Illinois Fighting Illini',
    'Illinois St': 'Illinois State Redbirds',
    'Incarnate Word': 'Incarnate Word Cardinals',
    'Indiana': 'Indiana Hoosiers',
    'Indiana St': 'Indiana State Sycamores',
    'Iona': 'Iona Gaels',
    'Iowa': 'Iowa Hawkeyes',
    'Iowa St': 'Iowa State Cyclones',
    'Jackson St': 'Jackson State Tigers',
    'Jacksonville': 'Jacksonville Dolphins',
    'Jacksonville St': 'Jacksonville State Gamecocks',
    'James Madison': 'James Madison Dukes',
    'Kansas': 'Kansas Jayhawks',
    'Kansas St': 'Kansas State Wildcats',
    'Kennesaw': 'Kennesaw State Owls',
    'Kent': 'Kent State Golden Flashes',
    'Kentucky': 'Kentucky Wildcats',
    'LSU': 'Louisiana State Fighting Tigers',
    'La Salle': 'La Salle Explorers',
    'Lafayette': 'Lafayette Leopards',
    'Lamar': 'Lamar Cardinals',
    'Lehigh': 'Lehigh Mountain Hawks',
    'Liberty': 'Liberty Flames',
    'Lipscomb': 'Lipscomb Bisons',
    'Long Beach St': 'Long Beach State 49ers',
    'Long Island': 'Long Island University Blackbirds',
    'Longwood': 'Longwood Lancers',
    'Louisiana Tech': 'Louisiana Tech Bulldogs',
    'Louisville': 'Louisville Cardinals',
    'Loy Marymount': 'Loyola Marymount Lions',
    'Loyola MD': 'Loyola (MD) Greyhounds',
    'Loyola-Chicago': 'Loyola (IL) Ramblers',
    'MA Lowell': 'Massachusetts-Lowell River Hawks',
    'MD E Shore': 'Maryland-Eastern Shore Hawks',
    'MS Valley St': 'Mississippi Valley State Delta Devils',
    'MTSU': 'Middle Tennessee Blue Raiders',
    'Maine': 'Maine Black Bears',
    'Manhattan': 'Manhattan Jaspers',
    'Marist': 'Marist Red Foxes',
    'Marquette': 'Marquette Golden Eagles',
    'Marshall': 'Marshall Thundering Herd',
    'Maryland': 'Maryland Terrapins',
    'Massachusetts': 'Massachusetts Minutemen',
    'McNeese St': 'McNeese State Cowboys',
    'Memphis': 'Memphis Tigers',
    'Mercer': 'Mercer Bears',
    'Miami FL': 'Miami (FL) Hurricanes',
    'Miami OH': 'Miami (OH) RedHawks',
    'Michigan': 'Michigan Wolverines',
    'Michigan St': 'Michigan State Spartans',
    'WI Milwaukee': 'Milwaukee Panthers',
    'Minnesota': 'Minnesota Golden Gophers',
    'Mississippi': 'Mississippi Rebels',
    'Mississippi St': 'Mississippi State Bulldogs',
    'Missouri': 'Missouri Tigers',
    'Missouri KC': 'Missouri-Kansas City Kangaroos',
    'Missouri St': 'Missouri State Bears',
    'Monmouth NJ': 'Monmouth Hawks',
    'Montana': 'Montana Grizzlies',
    'Montana St': 'Montana State Bobcats',
    'Morehead St': 'Morehead State Eagles',
    'Morgan St': 'Morgan State Bears',
    'Morris Brown': None, #u'Morris Brown Wolverines',
    "Mt St Mary's": "Mount St. Mary's Mountaineers",
    'Murray St': 'Murray State Racers',
    'N Colorado': 'Northern Colorado Bears',
    'N Dakota St': 'North Dakota State Bison',
    'N Illinois': 'Northern Illinois Huskies',
    'N Kentucky': 'Northern Kentucky Norse',
    'NC A&T': 'North Carolina A&T Aggies',
    'NC Central': 'North Carolina Central Eagles',
    'NC State': 'North Carolina State Wolfpack',
    'NE Illinois': None, #u'Northeastern Illinois Golden Eagles',
    'NE Omaha': 'Nebraska-Omaha Mavericks',
    'NJIT': 'NJIT Highlanders',
    'Navy': 'Navy Midshipmen',
    'Nebraska': 'Nebraska Cornhuskers',
    'Nevada': 'Nevada Wolf Pack',
    'New Hampshire': 'New Hampshire Wildcats',
    'New Mexico': 'New Mexico Lobos',
    'New Mexico St': 'New Mexico State Aggies',
    'New Orleans': 'New Orleans Privateers',
    'Niagara': 'Niagara Purple Eagles',
    'Nicholls St': 'Nicholls State Colonels',
    'Norfolk St': 'Norfolk State Spartans',
    'North Carolina': 'North Carolina Tar Heels',
    'North Dakota': 'North Dakota UND',
    'North Florida': 'North Florida Ospreys',
    'North Texas': 'North Texas Mean Green',
    'Northeastern': 'Northeastern Huskies',
    'Northern Arizona': 'Northern Arizona Lumberjacks',
    'Northern Iowa': 'Northern Iowa Panthers',
    'Northwestern': 'Northwestern Wildcats',
    'Northwestern LA': 'Northwestern State Demons',
    'Notre Dame': 'Notre Dame Fighting Irish',
    'Oakland': 'Oakland Golden Grizzlies',
    'Ohio': 'Ohio Bobcats',
    'Ohio St': 'Ohio State Buckeyes',
    'Okla City': None, #u'Oklahoma City Chiefs',
    'Oklahoma': 'Oklahoma Sooners',
    'Oklahoma St': 'Oklahoma State Cowboys',
    'Old Dominion': 'Old Dominion Monarchs',
    'Oral Roberts': 'Oral Roberts Golden Eagles',
    'Oregon': 'Oregon Ducks',
    'Oregon St': 'Oregon State Beavers',
    'Pacific': 'Pacific Tigers',
    'Penn': 'Pennsylvania Quakers',
    'Penn St': 'Penn State Nittany Lions',
    'Pepperdine': 'Pepperdine Waves',
    'Pittsburgh': 'Pittsburgh Panthers',
    'Portland': 'Portland Pilots',
    'Portland St': 'Portland State Vikings',
    'Prairie View': 'Prairie View Panthers',
    'Presbyterian': 'Presbyterian Blue Hose',
    'Princeton': 'Princeton Tigers',
    'Providence': 'Providence Friars',
    'Purdue': 'Purdue Boilermakers',
    'Quinnipiac': 'Quinnipiac Bobcats',
    'Radford': 'Radford Highlanders',
    'Rhode Island': 'Rhode Island Rams',
    'Rice': 'Rice Owls',
    'Richmond': 'Richmond Spiders',
    'Rider': 'Rider Broncs',
    'Robert Morris': 'Robert Morris Colonials',
    'Rutgers': 'Rutgers Scarlet Knights',
    'S Carolina St': 'South Carolina State Bulldogs',
    'S Dakota St': 'South Dakota State Jackrabbits',
    'S Illinois': 'Southern Illinois Salukis',
    'SC Upstate': 'South Carolina Upstate Spartans',
    'SE Louisiana': 'Southeastern Louisiana Lions',
    'SE Missouri St': 'Southeast Missouri State Redhawks',
    'SF Austin': 'Stephen F. Austin Lumberjacks',
    'SMU': 'Southern Methodist Mustangs',
    'Sacred Heart': 'Sacred Heart Pioneers',
    'Sam Houston St': 'Sam Houston State Bearkats',
    'Samford': 'Samford Bulldogs',
    'UT San Antonio': 'Texas-San Antonio Roadrunners',
    'San Diego': 'San Diego Toreros',
    'San Diego St': 'San Diego State Aztecs',
    'San Francisco': 'San Francisco Dons',
    'San Jose St': 'San Jose State Spartans',
    'Santa Barbara': 'UC-Santa Barbara Gauchos',
    'Santa Clara': 'Santa Clara Broncos',
    'Savannah St': 'Savannah State Tigers',
    'Seattle': 'Seattle Redhawks',
    'Seton Hall': 'Seton Hall Pirates',
    'Siena': 'Siena Saints',
    'South Alabama': 'South Alabama Jaguars',
    'South Carolina': 'South Carolina Gamecocks',
    'South Dakota': 'South Dakota Coyotes',
    'South Florida': 'South Florida Bulls',
    'Southern Miss': 'Southern Mississippi Golden Eagles',
    'Southern Univ': 'Southern Jaguars',
    'Southern Utah': 'Southern Utah Thunderbirds',
    'St Bonaventure': 'St. Bonaventure Bonnies',
    'St Francis NY': 'St. Francis (NY) Terriers',
    'St Francis PA': 'Saint Francis (PA) Red Flash',
    "St John's": "St. John's (NY) Red Storm",
    "St Joseph's PA": "Saint Joseph's Hawks",
    'St Louis': 'Saint Louis Billikens',
    "St Mary's CA": "Saint Mary's (CA) Gaels",
    "St Peter's": "Saint Peter's Peacocks",
    'Stanford': 'Stanford Cardinal',
    'Stetson': 'Stetson Hatters',
    'Stony Brook': 'Stony Brook Seawolves',
    'Syracuse': 'Syracuse Orange',
    'TAM C. Christi': 'Texas A&M-Corpus Christi Islanders',
    'TCU': 'Texas Christian Horned Frogs',
    'TN Martin': 'Tennessee-Martin Skyhawks',
    'TX Pan American': 'Texas-Rio Grande Valley Vaqueros',
    'TX Southern': 'Texas Southern Tigers',
    'Temple': 'Temple Owls',
    'Tennessee': 'Tennessee Volunteers',
    'Tennessee St': 'Tennessee State Tigers',
    'Tennessee Tech': 'Tennessee Tech Golden Eagles',
    'Texas': 'Texas Longhorns',
    'Texas A&M': 'Texas A&M Aggies',
    'Texas St': 'Texas State Bobcats',
    'Texas Tech': 'Texas Tech Red Raiders',
    'Toledo': 'Toledo Rockets',
    'Towson': 'Towson Tigers',
    'Troy': 'Troy Trojans',
    'Tulane': 'Tulane Green Wave',
    'Tulsa': 'Tulsa Golden Hurricane',
    'UAB': 'Alabama-Birmingham Blazers',
    'UC Davis': 'UC-Davis Aggies',
    'UC Irvine': 'UC-Irvine Anteaters',
    'UC Riverside': 'UC-Riverside Highlanders',
    'UCF': 'Central Florida Knights',
    'UCLA': 'UCLA Bruins',
    'ULL': None,
    'ULM': 'Louisiana-Monroe Warhawks',
    'UMBC': 'Maryland-Baltimore County Retrievers',
    'UNC Asheville': 'North Carolina-Asheville Bulldogs',
    'UNC Greensboro': 'North Carolina-Greensboro Spartans',
    'UNC Wilmington': 'North Carolina-Wilmington Seahawks',
    'UNLV': 'Nevada-Las Vegas Rebels',
    'USC': 'South Carolina Upstate Spartans',
    'UTEP': 'Texas-El Paso Miners',
    'Utah': 'Utah Utes',
    'Utah St': 'Utah State Aggies',
    'Utah Valley': 'Utah Valley Wolverines',
    'Utica': None,
    'VA Commonwealth': 'Virginia Commonwealth Rams',
    'VMI': 'Virginia Military Institute Keydets',
    'Valparaiso': 'Valparaiso Crusaders',
    'Vanderbilt': 'Vanderbilt Commodores',
    'Vermont': 'Vermont Catamounts',
    'Villanova': 'Villanova Wildcats',
    'Virginia': 'Virginia Cavaliers',
    'Virginia Tech': 'Virginia Tech Hokies',
    'Wagner': 'Wagner Seahawks',
    'Wake Forest': 'Wake Forest Demon Deacons',
    'Washington': 'Washington Huskies',
    'Washington St': 'Washington State Cougars',
    'Weber St': 'Weber State Wildcats',
    'West Virginia': 'West Virginia Mountaineers',
    'W Carolina': 'Western Carolina Catamounts',
    'W Illinois': 'Western Illinois Leathernecks',
    'W Kentucky': 'Western Kentucky Hilltoppers',
    'W Michigan': 'Western Michigan Broncos',
    'W Salem St': None,
    'W Texas A&M': None, #u'West Texas A&M Buffaloes',
    'Wichita St': 'Wichita State Shockers',
    'William & Mary': 'William & Mary Tribe',
    'Winthrop': 'Winthrop Eagles',
    'Wisconsin': 'Wisconsin Badgers',
    'Wofford': 'Wofford Terriers',
    'Wright St': 'Wright State Raiders',
    'Wyoming': 'Wyoming Cowboys',
    'Xavier': 'Xavier Musketeers',
    'Yale': 'Yale Bulldogs',
    'Youngstown St': 'Youngstown State Penguins'}