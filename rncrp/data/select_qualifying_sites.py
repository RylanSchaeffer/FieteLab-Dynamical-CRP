import pandas as pd


#################################################################
# Obtain list of sites with data from 1946 to END_YEAR
#################################################################
def satisfy_dates(
        inventory_loc: str = "/om2/user/gkml/FieteLab-Recursive-Nonstationary-CRP/exp2_climate/ghcnd-inventory.txt",
        end_year: int = 2020):

    sites = pd.read_csv(inventory_loc, sep='\s+', header=None)
    sites.columns = ["station", "lat", "long", "field", "start", "end"]

    sites = sites[(sites.start <= 1946) & (sites.end >= end_year)]
    sites_to_fetch = sites.station.unique().tolist()
    print("Number of sites satisfying date range:", len(sites_to_fetch))

    with open("/om2/user/gkml/FieteLab-Recursive-Nonstationary-CRP/exp2_climate/sites_with_valid_dates_" + str(
            end_year) + ".txt", "w+") as f:
        for item in sites_to_fetch:
            f.write(item + "\n")
    f.close()


#################################################################
# Filter for stations with sufficient data (tmin, tmax, pcpn)
#################################################################
def qualify_checker(df):
    col_names = df.columns
    if 'TMIN' not in col_names or 'TMAX' not in col_names or 'PRCP' not in col_names:
        return False

    min_required_days = {1: 28, 2: 26, 3: 28, 4: 27, 5: 28, 6: 27, 7: 28, 8: 28, 9: 27, 10: 28, 11: 27, 12: 28}
    max_nulls = {1: 3, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3, 11: 3, 12: 3}

    df['YEAR'] = df.DATE.apply(lambda x: int(x[:4]))
    df['MONTH'] = df.DATE.apply(lambda x: int(x[5:7]))

    for year in range(1946, end_year + 1):
        for month in range(1, 13):
            # At least 90% data present per month (and thus per year)
            if df[(df.YEAR == year) & (df.MONTH == month)].TMIN.count().sum() < min_required_days[month]:
                return False
            if df[(df.YEAR == year) & (df.MONTH == month)].TMAX.count().sum() < min_required_days[month]:
                return False
            if df[(df.YEAR == year) & (df.MONTH == month)].PRCP.count().sum() < min_required_days[month]:
                return False
            # No more than 10% of data missing per month (and thus per year)
            if df[(df.YEAR == year) & (df.MONTH == month)].TMIN.isnull().sum() > max_nulls[month]:
                return False
            if df[(df.YEAR == year) & (df.MONTH == month)].TMAX.isnull().sum() > max_nulls[month]:
                return False
            if df[(df.YEAR == year) & (df.MONTH == month)].PRCP.isnull().sum() > max_nulls[month]:
                return False
    return True


def get_qualifying_sites(end_year: int = 2020):
    qualifying_sites = []
    qualifying_site_links = []
    with open('/om2/user/gkml/FieteLab-Recursive-Nonstationary-CRP/exp2_climate/sites_with_valid_dates_' + str(
            end_year) + '.txt') as file:
        for site_name in file:
            site_csv_path = '/om2/user/gkml/FieteLab-Recursive-Nonstationary-CRP/exp2_climate/data/' + site_name.strip() + '.csv'
            site_csv_link = 'https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/' + site_name.strip() + '.csv'

            # Load file into dataframe
            df = pd.read_csv(site_csv_path, low_memory=False)
            # df = pd.read_csv(site_csv, compression='gzip',low_memory=False) # if .gz version downloaded

            # If criteria satisfied
            if qualify_checker(df):
                qualifying_sites.append(site_csv_path)
                qualifying_site_links.append(site_csv_link)
                print(site_name)

    print("Number of qualifying sites:", len(qualifying_sites))
    with open("/om2/user/gkml/FieteLab-Recursive-Nonstationary-CRP/exp2_climate/qualifying_sites_" + str(
            end_year) + ".txt", "w+") as f:
        for site_csv_path in qualifying_sites:
            f.write(site_csv_path + "\n")
    f.close()

    with open("/om2/user/gkml/FieteLab-Recursive-Nonstationary-CRP/exp2_climate/qualifying_site_links_" + str(
            end_year) + ".txt", "w+") as f:
        for site_csv_link in qualifying_site_links:
            f.write(site_csv_link + "\n")
    f.close()


end_year = 2015
# satisfy_dates(end_year=end_year)
get_qualifying_sites(end_year)
