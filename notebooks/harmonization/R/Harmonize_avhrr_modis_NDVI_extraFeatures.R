library(stars); library(tidyverse); library(data.table); library(lubridate)
library(dtplyr, warn.conflicts = FALSE)
setDTthreads(threads=0)
library(mgcv)
library(RcppArmadillo)

# These AVHRR datasets have already been filtered/cleaned in a python script,
# so the usual data-table filtering has been removed in this script.
avhrr_path = '/g/data/os22/chad_tmp/climate-carbon-interactions/data/NDVI_harmonization/AVHRR_5km_monthly_1982_2013_extraFeatures.nc'
modis_path = '/g/data/os22/chad_tmp/climate-carbon-interactions/data/NDVI_harmonization/MODIS_NDVI_5km_monthly_200003_202212.nc'

## Read AVHRR data 'bands' separately
tmp_median = stars::read_ncdf(avhrr_path, var="NDVI_avhrr", make_time = T, proxy=F)
tmp_median <- tmp_median %>% set_names(c("ndvi_cdr"))

tmp_mod_mean = stars::read_ncdf(avhrr_path, var="NDVI_modis_mean", make_time = T, proxy=F)
tmp_mod_mean <- tmp_mod_mean %>% set_names(c("ndvi_modis_mean"))

tmp_rain_cml3 = stars::read_ncdf(avhrr_path, var="rain_cml3", make_time = T, proxy=F)
tmp_rain_cml3 <- tmp_rain_cml3 %>% set_names(c("rain_cml3"))

tmp_srad = stars::read_ncdf(avhrr_path, var="srad", make_time = T, proxy=F)
tmp_srad <- tmp_srad %>% set_names(c("srad"))

tmp_sza = stars::read_ncdf(avhrr_path, var='SZEN_median', make_time = T, proxy=F)
tmp_sza <- tmp_sza %>% set_names(c('sza'))

tmp_tod = stars::read_ncdf(avhrr_path, var='TIMEOFDAY_median', make_time = T, proxy=F)
tmp_tod <- tmp_tod %>% set_names(c('tod'))

## Convert AVHRR data into data tables, add 'month' var.
tmp <- c(tmp_median, tmp_sza, tmp_tod, tmp_mod_mean, tmp_rain_cml3, tmp_srad)
d_cdr <- tmp %>% units::drop_units() %>% as.data.frame() %>% as.data.table()
d_cdr <- d_cdr %>% mutate(month=month(time)) %>% as.data.table()

#clear up some memory
rm(tmp_median, tmp_sza, tmp_tod,
   tmp, tmp_mod_mean, tmp_rain_cml3, tmp_srad)
gc()

# Import MODIS NDVI--------------------------------------------------------
tmp_nm <- stars::read_ncdf(modis_path, var="NDVI_median", make_time = T, proxy=F)
tmp_nm <- tmp_nm %>% set_names(c('ndvi_mcd'))

d_mcd <- tmp_nm %>% units::drop_units() %>% as_tibble(tmp_nm) %>% as.data.table()
d_mcd <- d_mcd[between(time,ymd("2000-03-01"),ymd("2013-12-31"))==T]
rm(tmp_nm)
gc()

# Calibrate AVHRR CDR to approximate MODIS NDVI -----------------------------------------

# merge the modis and avhrr datasets
dc2 <- merge(d_mcd[,.(longitude,latitude,time,ndvi_mcd)],
           d_cdr[,.(longitude,latitude,time,ndvi_cdr,
                    sza,ndvi_modis_mean,rain_cml3,srad)], all=TRUE,
           by=c("longitude","latitude","time"))

dc2 <- dc2[,`:=`(month=month(time))]
rm(d_mcd)
rm(d_cdr)
gc()

# training and testing samples
dc2_train <- dc2[is.na(ndvi_mcd)==F][between(time,ymd("2000-03-01"),ymd("2013-12-31"))==T][sample(.N, 1e6)]
dc2_test <- dc2[is.na(ndvi_mcd)==F][between(time,ymd("2000-03-01"),ymd("2013-12-31"))==T][sample(.N, 1e6)]

# Train a GAM on the datasets
mc10 <- bam(ndvi_mcd ~ s(ndvi_cdr,bs='cs')+
            s(month,bs='cc')+
            s(sza,bs='cs')+
            #s(tod,bs='cs')+ #hashed out trying to reduce memory
            s(ndvi_modis_mean)+
            s(rain_cml3,bs='cr', k=30)+
            s(srad,bs='cr', k=30)+
            te(longitude,latitude),
            data=dc2_train[ndvi_mcd>0.1], 
          discrete=T, 
          select=T)

summary(mc10)

# predict on the testing data
summary(predict(mc10, newdata=dc2_test[is.na(ndvi_cdr)==F]))

# Correlation on the testing data 
cor(predict(mc10, newdata=dc2_test[is.na(ndvi_cdr)==F]),
dc2_test[is.na(ndvi_cdr)==F]$ndvi_mcd)**2

# Apply Calibration prediction in parallel --------------------------------

library(foreach); library(doParallel)
n_cores <- 8
cl <- makeCluster(n_cores, outfile="")

#registerDoParallel(cl)
registerDoSEQ() #use this to run sequential for debugging

dc2 <- mutate(dc2, year=data.table::year(time),
            month=data.table::month(time)) %>% 
            as.data.table()

vec_years <- 1982:2013

out <- foreach(i = 1:length(vec_years), 
             .packages = c("mgcv","data.table","tidyverse",
                           "dtplyr"),
             .combine=rbind) %dopar% {
               out <- dc2[year==vec_years[i]][is.na(ndvi_cdr)==F] %>% 
                 mutate(ndvi_mcd_pred = predict(mc10, 
                                                newdata=., 
                                                newdata.guaranteed = T,
                                                discrete = TRUE)) %>% 
                 as.data.table() # force computation
               out
             } 

## ---prepare for export-------
d_export <- merge(dc2,
                out[, .(longitude,latitude, time, 
                        ndvi_mcd_pred)],
                by=c("longitude","latitude",'time'),
                all=TRUE)

tmp <- st_as_stars(d_export, dims = c("longitude","latitude","time"))

## requires stars 0.6-1 or greater
stars::write_mdim(tmp,
         paste('/g/data/os22/chad_tmp/climate-carbon-interactions/data/NDVI_harmonization/regions/Harmonized_GAM_NDVI_AVHRR_MODIS_1982_2013.nc'),
         layer = c("ndvi_mcd", "ndvi_cdr", "ndvi_mcd_pred", "month", "year"
         ))


