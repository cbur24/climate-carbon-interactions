library(stars); library(tidyverse); library(data.table); library(lubridate)
library(dtplyr, warn.conflicts = FALSE)
setDTthreads(threads=0)
library(mgcv)
library(RcppArmadillo)

avhrr_path = '/g/data/os22/chad_tmp/climate-carbon-interactions/data/TAS_AVHRR_5km_monthly_1982_2013.nc'
modis_path = '/g/data/os22/chad_tmp/climate-carbon-interactions/data/TAS_MODIS_NDVI_median_5km_monthly_2001_2022.nc'

## Read AVHRR data
tmp_median = stars::read_ncdf(avhrr_path, var="NDVI_median", make_time = T, proxy=F)
tmp_median <- tmp_median %>% set_names(c(paste(var, "cdr", sep="_")))

tmp_nobs = stars::read_ncdf(avhrr_path, var='n_obs', make_time = T, proxy=F)
tmp_nobs <- tmp_nobs %>% set_names(c('nobs'))

tmp_sd = stars::read_ncdf(avhrr_path, var="NDVI_stddev", make_time = T, proxy=F)
tmp_sd <- tmp_sd %>% set_names(c('sd'))

tmp_sza = stars::read_ncdf(avhrr_path, var='SZEN_median', make_time = T, proxy=F)
tmp_sza <- tmp_sza %>% set_names(c('sza'))

tmp_tod = stars::read_ncdf(avhrr_path, var='TIMEOFDAY_median', make_time = T, proxy=F)
tmp_tod <- tmp_tod %>% set_names(c('tod'))

## Convert AVHRR data into data tables
tmp <- c(tmp_median, tmp_nobs, tmp_sd, tmp_sza, tmp_tod)
d_cdr <- tmp %>% units::drop_units() %>% as.data.frame() %>% as.data.table()
d_cdr <- d_cdr %>% mutate(month=month(time))

#clear up some memory
rm(tmp_median, tmp_nobs, tmp_sd, tmp_sza, tmp_tod, tmp)
gc()

## calculate climatologies
d_cdr_norms <- d_cdr %>% 
  lazy_dt() %>%
  mutate(month=month(time)) %>% 
  group_by(x,y,month) %>% 
  summarize(ndvi_u = mean(ndvi_cdr,na.rm=TRUE), 
            sza_u = mean(sza,na.rm=TRUE), 
            tod_u = mean(tod, na.rm=TRUE), 
            ndvi_sd = sd(ndvi_cdr, na.rm=TRUE), 
            sza_sd = sd(sza,na.rm=TRUE), 
            tod_sd = sd(tod,na.rm=TRUE)) %>% 
  ungroup() %>% 
  as.data.table()

# calculate anomalies
d_cdr <- merge(d_cdr_norms, d_cdr, by=c("x","y","month"))
d_cdr <- d_cdr %>% 
  lazy_dt() %>% 
  mutate(ndvi_anom = ndvi_cdr - ndvi_u, 
         sza_anom = sza - sza_u, 
         tod_anom = tod - tod_u) %>% 
  mutate(ndvi_anom_sd = ndvi_anom/ndvi_sd, 
         sza_anom_sd = sza_anom/sza_sd, 
         tod_anom_sd = tod_anom/tod_sd) %>% 
  as.data.table()

gc()

#Filter the dataset by climatolgies noise etc
d_cdr <- d_cdr %>% 
  lazy_dt() %>% 
  filter(nobs >= 3) %>% 
  filter(ndvi_cdr >= 0.1) %>% 
  mutate(cv_cdr = sd/ndvi_cdr) %>% 
  filter(cv_cdr < 0.25) %>% 
  filter(between(ndvi_anom_sd, -3.5,3.5)) %>% 
  filter(between(sza_anom_sd, -3.5,3.5)) %>% 
  filter(between(tod_anom_sd, -3.5,3.5)) %>% 
  as.data.table()

rm(d_cdr_norms)
gc()

# Import MCD43 NDVI--------------------------------------------------------
tmp_nm <- stars::read_ncdf(modis_path, var="LST_median", make_time = T, proxy=F)
tmp_nm <- tmp_nm %>% set_names(c('NDVI_mcd'))

d_mcd <- tmp_nm %>% units::drop_units() %>% as_tibble(tmp_nm) %>% as.data.table()
d_mcd <- d_mcd[between(time,ymd("2001-01-01"),ymd("2013-12-31"))==T]
rm(tmp_nm)
gc()

# Calibrate AVHRR CDR to approximate MCD43 NDVI -----------------------------------------

# merge the modis and avhrr datasets
dc2 <- merge(d_mcd[,.(x,y,time,ndvi_mcd)],
             d_cdr[,.(x,y,time,ndvi_cdr,sza,tod)], all=TRUE, 
             by=c("x","y","time"))

dc2 <- dc2[,`:=`(month=month(time))]
rm(d_mcd)
rm(d_cdr)
gc()

# training and testing samples
dc2_train <- dc2[is.na(ndvi_mcd)==F][between(time,ymd("2001-01-01"),ymd("2013-12-31"))==T][sample(.N, 1e6)]
dc2_test <- dc2[is.na(ndvi_mcd)==F][between(time,ymd("2001-01-01"),ymd("2013-12-31"))==T][sample(.N, 1e6)]

# Train a GAM on the datasets
mc10 <- bam(ndvi_mcd ~ s(ndvi_cdr,bs='cs')+
              s(month,bs='cc')+
              s(sza,bs='cs')+
              s(tod,bs='cs')+
              te(x,y), data=dc2_train[ndvi_mcd>0.1], 
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
                  out[, .(x, y, time, 
                          ndvi_mcd_pred)],
                  by=c("x",'y','time'),
                  all=TRUE)

tmp <- st_as_stars(d_export, dims = c("x","y","time"))

## requires stars 0.6-1 or greater
stars::write_mdim(tmp, 
           '/g/data/os22/chad_tmp/climate-carbon-interactions/data/TAS_Harmonized_NDVI_AVHRR_MODIS_1982_2013.nc', 
           layer = c("ndvi_mcd", "ndvi_cdr", "ndvi_mcd_pred", "month", "year"
           )) 



