rm(list = ls()) # clear the memory
cat("\014")  # clear console

library(MacrobondAPI)
library(timeseriesdb)
library(purrr)
library(tsbox)

mb_keys <- read.table("data_eu_mb_in.csv", header = FALSE)$V1

seriesRequest <- CreateUnifiedTimeSeriesRequest()
setFrequency(seriesRequest, "Monthly")
setStartDate(seriesRequest, "1975-01-01")

for (ser in mb_keys) {
  addSeries(seriesRequest, ser)
}

res <- MacrobondAPI::FetchTimeSeries(seriesRequest)
errs <- keep(res, getIsError)

# Show number of errors by type (e.g. "not permitted")
map_chr(errs, getErrorMessage) %>% table()

# Discard results with errors
res <- discard(res, getIsError)

# Set prefix for DB keys
db_keys <- names(res)

# Load series data
tsl <- imap(res, function(x, y) {
  cat(y, "\n")
  vals <- MacrobondAPI::getValues(x)
  vals[is.nan(vals)] <- NA
  dates <- MacrobondAPI::getDatesAtStartOfPeriod(x)
  xts(vals, order.by = dates)
})
names(tsl) <- tolower(db_keys)

# Load series titles
titles <- data.frame(ts_key = tolower(db_keys), title = map_chr(res, MacrobondAPI::getTitle))

write.table(titles, "keys_titles.csv", sep = ";")
write.table(tsl, "data_eu.csv", sep = ";")
