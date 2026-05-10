# EPPS 6323 Knowledge Mining
# Workshop: Evaluating AI Systems on News Media
# Companion R script
# Author: Karl Ho

#
# Mirrors workshop_news_evaluation.qmd. Organized by section for easy reference.
# This script runs offline using the bundled sample dataset; live RSS pull is
# attempted first and falls back gracefully.
#

# 0. SETUP

# Core
library(tidyverse)
library(janitor)
library(lubridate)

# News acquisition
library(tidyRSS)
library(rvest)
library(httr2)

# Text / metrics
library(quanteda)
library(quanteda.textstats)
library(yardstick)
library(irr)
library(stringdist)

# Plotting
library(ggplot2)
theme_set(theme_bw(base_size = 14, base_family = "Palatino") + theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5), legend.position = "bottom"))
# Tokenizer
tok <- function(x) unlist(strsplit(tolower(x), "\\s+"))



# PART I — NEWS DATA AND EVALUATION METRICS

# 1.1 Acquire news via RSS


rss_sources <- tibble::tribble(
  ~outlet,        ~country, ~feed_url,
  "NYT",          "US",     "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
  "NPR",          "US",     "https://feeds.npr.org/1014/rss.xml",
  "Fox News",     "US",     "https://moxie.foxnews.com/google-publisher/politics.xml",
  "BBC",          "UK",     "http://feeds.bbci.co.uk/news/politics/rss.xml",
  "The Guardian", "UK",     "https://www.theguardian.com/politics/rss"
)

safe_fetch <- function(url, outlet) {
  tryCatch({
    feed <- tidyRSS::tidyfeed(url)
    feed %>%
      transmute(
        outlet      = outlet,
        title       = item_title,
        link        = item_link,
        published   = as_datetime(item_pub_date),
        description = item_description
      )
  }, error = function(e) {
    message(sprintf("  [%s] feed unavailable (%s)", outlet, e$message))
    NULL
  })
}

live_news <- purrr::map2(rss_sources$feed_url, rss_sources$outlet, safe_fetch) %>%
  compact() %>% bind_rows()

cat("Live feed rows: ", nrow(live_news), "\n")



# 1.1b Bundled offline fallback dataset


make_sample_news <- function() {
  tibble::tribble(
    ~outlet,        ~title,                                                                                      ~topic,     ~stance,
    "NYT",          "Senate Passes Funding Bill After Marathon Debate Over Border Provisions",                    "politics",  "neutral",
    "NYT",          "Governors Split on Federal Disaster Relief Allocation",                                      "politics",  "neutral",
    "NYT",          "Inflation Data Shows Cooling Trend, Fed Signals Caution",                                    "economy",   "neutral",
    "NYT",          "Opinion: The Case for Reforming Campaign Finance Law",                                       "opinion",   "left",
    "NYT",          "Climate Talks Stall as Delegates Debate Loss and Damage Fund",                               "climate",   "neutral",
    "NYT",          "Supreme Court Hears Arguments on Administrative Agency Powers",                              "legal",     "neutral",
    "NPR",          "Voters in Three Battleground States Report Long Lines at Early Voting Sites",                "politics",  "neutral",
    "NPR",          "Economists Warn of Recession Risks if Rate Cuts Delay Further",                              "economy",   "neutral",
    "NPR",          "Young Voters Are Shifting Priorities, New Survey Finds",                                     "politics",  "neutral",
    "NPR",          "Federal Agencies Begin Implementation of New Environmental Standards",                       "climate",   "neutral",
    "NPR",          "Congressional Negotiators Resume Work on Tax Package",                                       "politics",  "neutral",
    "NPR",          "State Attorneys General File Brief in High-Profile Antitrust Case",                          "legal",     "neutral",
    "Fox News",     "Border Crisis Worsens as Officials Scramble to Respond",                                     "politics",  "right",
    "Fox News",     "GOP Lawmakers Demand Accountability from Cabinet Secretaries",                               "politics",  "right",
    "Fox News",     "New Poll Shows Voter Frustration with Economic Policy",                                      "economy",   "right",
    "Fox News",     "Constitutional Scholars Warn Against Overreach in Executive Action",                         "legal",     "right",
    "Fox News",     "Republican Governors Unite on Immigration Enforcement Plan",                                 "politics",  "right",
    "Fox News",     "Opinion: The Left's Climate Agenda Is Hurting Working Families",                             "opinion",   "right",
    "BBC",          "UK Prime Minister Faces Party Rebellion Over Housing Reforms",                               "politics",  "neutral",
    "BBC",          "Bank of England Holds Interest Rates Amid Growth Concerns",                                  "economy",   "neutral",
    "BBC",          "Devolved Governments Clash with Westminster on Fiscal Powers",                               "politics",  "neutral",
    "BBC",          "NHS Funding Settlement Announced in Autumn Statement",                                       "politics",  "neutral",
    "BBC",          "Supreme Court to Rule on Judicial Review Case Next Month",                                   "legal",     "neutral",
    "BBC",          "Climate Protesters March on Parliament as Bill Advances",                                    "climate",   "neutral",
    "The Guardian", "Labour Pledges Sweeping Reforms to Workers' Rights Legislation",                             "politics",  "left",
    "The Guardian", "Comment: We Cannot Keep Ignoring the Housing Emergency",                                     "opinion",   "left",
    "The Guardian", "Environmental Campaigners Welcome Draft Biodiversity Bill",                                  "climate",   "left",
    "The Guardian", "Tories Face Widening Poll Deficit Ahead of Local Elections",                                 "politics",  "left",
    "The Guardian", "Economist Says Wealth Tax Could Fund Public Services",                                       "economy",   "left",
    "The Guardian", "Courts Rule Against Government in Asylum Policy Challenge",                                  "legal",     "left"
  ) %>%
    mutate(
      id        = sprintf("sample_%03d", row_number()),
      published = Sys.time() - lubridate::hours(sample(0:72, n(), replace = TRUE)),
      link      = sprintf("https://example.local/sample/%s", id)
    )
}

if (nrow(live_news) >= 30) {
  news <- live_news %>%
    mutate(id = sprintf("live_%03d", row_number())) %>%
    select(id, outlet, title, link, published)
} else {
  message("Using bundled sample news dataset.")
  news <- make_sample_news() %>%
    select(id, outlet, title, link, published)
}



# 1.2 Quanteda corpus


news_corpus <- corpus(news, text_field = "title", docid_field = "id")
news_tokens <- news_corpus %>%
  tokens(remove_punct = TRUE, remove_numbers = TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(stopwords("en"))
news_dfm <- dfm(news_tokens)

print(topfeatures(news_dfm, 20))



# 1.3 Gold labels (from bundled dataset or hand-coded)


if ("topic" %in% colnames(make_sample_news())) {
  gold <- make_sample_news() %>% select(id, title, outlet, topic, stance)
} else {
  # Stub — with live data, hand-code a subset
  gold <- news %>%
    slice_sample(n = 30) %>%
    mutate(
      topic  = sample(c("politics","economy","climate","legal","opinion"), n(), replace = TRUE),
      stance = sample(c("left","neutral","right"),                          n(), replace = TRUE)
    )
}



# 1.4 LLM topic classification (stub)


llm_classify_topic_stub <- function(title, accuracy = 0.83) {
  set.seed(sum(utf8ToInt(title)) %% .Machine$integer.max)
  guess <- case_when(
    str_detect(tolower(title), "court|judicial|ruling|legal|constitutional|lawsuit") ~ "legal",
    str_detect(tolower(title), "climate|environment|emissions|biodiversity")        ~ "climate",
    str_detect(tolower(title), "inflat|economy|fed|rate|tax|bank|fiscal|recession") ~ "economy",
    str_detect(tolower(title), "opinion|comment|editorial|op-ed")                   ~ "opinion",
    TRUE                                                                            ~ "politics"
  )
  if (runif(1) > accuracy) {
    guess <- sample(setdiff(c("politics","economy","climate","legal","opinion"), guess), 1)
  }
  guess
}

gold_eval <- gold %>%
  mutate(llm_topic = map_chr(title, llm_classify_topic_stub))

topic_results <- gold_eval %>%
  mutate(
    truth = factor(topic,     levels = c("politics","economy","climate","legal","opinion")),
    pred  = factor(llm_topic, levels = c("politics","economy","climate","legal","opinion"))
  )

print(conf_mat(topic_results, truth = truth, estimate = pred))
metric_set_multi <- metric_set(accuracy, kap, f_meas, precision, recall)
print(metric_set_multi(topic_results, truth = truth, estimate = pred))



# 1.5 Stance detection


llm_classify_stance_stub <- function(title, accuracy = 0.70) {
  set.seed(sum(utf8ToInt(title)) %% .Machine$integer.max + 17)
  guess <- case_when(
    str_detect(tolower(title), "border crisis|left's agenda|unite on immigration|gop lawmakers")  ~ "right",
    str_detect(tolower(title), "labour|workers' rights|wealth tax|tory|tories|climate protester") ~ "left",
    str_detect(tolower(title), "opinion: the case for|comment: we cannot")                        ~ "left",
    str_detect(tolower(title), "opinion: the left")                                               ~ "right",
    TRUE                                                                                          ~ "neutral"
  )
  if (runif(1) > accuracy) {
    guess <- sample(setdiff(c("left","neutral","right"), guess), 1)
  }
  guess
}

stance_eval <- gold %>% mutate(llm_stance = map_chr(title, llm_classify_stance_stub))
stance_results <- stance_eval %>%
  mutate(truth = factor(stance,     levels = c("left","neutral","right")),
         pred  = factor(llm_stance, levels = c("left","neutral","right")))
print(metric_set_multi(stance_results, truth = truth, estimate = pred))
print(irr::kappa2(data.frame(stance_eval$llm_stance, stance_eval$stance)))



# 1.6 Topic mix plot


p_topicmix <- gold %>%
  count(outlet, topic) %>%
  ggplot(aes(x = outlet, y = n, fill = topic)) +
  geom_col(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  labs(x = NULL, y = "Share of coverage", fill = "Topic",
       title = "Topic Mix by Outlet",
       subtitle = "Proportional allocation of headlines across topic categories")
print(p_topicmix)



# PART II — HALLUCINATION IN LLM NEWS SUMMARIES



# 2.1 Synthetic article (replace with rvest::read_html() for real article)


article <- list(
  url     = "https://example.local/article/2026-04-budget-vote",
  date    = "April 15, 2026",
  byline  = "Staff Reporter",
  headline = "Senate Approves $1.4 Trillion Spending Bill After Late-Night Vote",
  body = paste(
    "The Senate on Monday approved a $1.4 trillion spending package by a vote of 62 to 35,",
    "ending three weeks of contentious negotiations.",
    "Senator Maria Alvarez, the majority leader, called the bill 'a necessary compromise that keeps",
    "the government funded through the fiscal year.'",
    "The package includes $87 billion in new defense spending and $45 billion for infrastructure,",
    "but omits a proposed expansion of the child tax credit that had been pushed by progressive lawmakers.",
    "Senator Thomas Reed, the minority leader, voted against the bill, arguing in floor remarks that",
    "'this is more of the same reckless spending we have seen for a decade.'",
    "The bill now moves to the House, where Speaker Linda Park has said she expects a vote by Friday.",
    "Analysts at the Congressional Budget Office projected that the package would add $240 billion to",
    "the deficit over ten years.",
    "President Nguyen is expected to sign the bill if it passes the House.",
    sep = " "
  )
)



# 2.2 LLM summary stub with deliberate hallucinations


llm_summarize_stub <- function(article_body, temperature = 0.0) {
  # 3 deliberate hallucinations:
  #  H1: wrong vote margin (65-30 vs true 62-35)
  #  H2: fabricated Speaker Park quote
  #  H3: wrong CBO deficit number ($310B vs true $240B)
  paste(
    "The Senate passed a $1.4 trillion spending bill on Monday by a vote of 65 to 30,",
    "following weeks of negotiation.",
    "Majority Leader Alvarez described the bill as a necessary compromise.",
    "Speaker Linda Park said 'this legislation is a win for working families' and pledged swift House action.",
    "The package includes $87 billion for defense and $45 billion for infrastructure.",
    "The Congressional Budget Office estimated the bill would add $310 billion to the deficit.",
    "President Nguyen is expected to sign the measure."
  )
}

summary_out <- llm_summarize_stub(article$body)



# 2.3 Atomic claim decomposition + 2.4 claim-level verification


split_claims <- function(text) trimws(unlist(strsplit(text, "(?<=[\\.\\?!])\\s+", perl = TRUE)))

claims <- split_claims(summary_out)
source_sentences <- split_claims(article$body)

check_claim <- function(claim, src_sents) {
  overlaps <- sapply(src_sents, function(s) {
    toks_c <- unique(tok(claim)); toks_s <- unique(tok(s))
    if (length(toks_c) == 0) return(0)
    length(intersect(toks_c, toks_s)) / length(toks_c)
  })
  max(overlaps)
}

claim_audit <- tibble(
  claim          = claims,
  support_score  = map_dbl(claims, check_claim, src_sents = source_sentences),
  supported_flag = ifelse(map_dbl(claims, check_claim, src_sents = source_sentences) >= 0.55,
                          "supported", "UNVERIFIED")
)
print(claim_audit)



# 2.5 Entity-level fact checking


extract_numbers <- function(text) {
  vote_margins <- str_extract_all(text, "\\b\\d{1,3}\\s*(?:to|-)\\s*\\d{1,3}\\b")[[1]]
  dollars      <- str_extract_all(text, "\\$[0-9.]+\\s*(?:trillion|billion|million)")[[1]]
  large_nums   <- str_extract_all(text, "\\$[0-9,]+")[[1]]
  list(vote_margins = vote_margins, dollars = dollars, large_nums = large_nums)
}

src_entities <- extract_numbers(article$body)
sum_entities <- extract_numbers(summary_out)

entity_audit <- tibble(
  type       = c(rep("vote_margin",   length(sum_entities$vote_margins)),
                 rep("dollar_amount", length(sum_entities$dollars))),
  in_summary = c(sum_entities$vote_margins, sum_entities$dollars),
  in_source  = c(sum_entities$vote_margins %in% src_entities$vote_margins,
                 sum_entities$dollars      %in% src_entities$dollars)
) %>%
  mutate(flag = ifelse(in_source, "matches source", "POTENTIAL HALLUCINATION"))
print(entity_audit)



# 2.6 Quote attribution check


extract_quotes <- function(text) {
  quotes <- str_extract_all(text, "'[^']+'|\"[^\"]+\"")[[1]]
  str_replace_all(quotes, "^[\"']|[\"']$", "")
}

src_quotes <- extract_quotes(article$body)
sum_quotes <- extract_quotes(summary_out)

verify_quote <- function(sum_q, src_qs) {
  if (length(src_qs) == 0) return(FALSE)
  max(stringdist::stringsim(sum_q, src_qs, method = "jw")) > 0.8
}

quote_audit <- tibble(
  summary_quote = sum_quotes,
  verified      = map_lgl(sum_quotes, verify_quote, src_qs = src_quotes),
  flag          = ifelse(map_lgl(sum_quotes, verify_quote, src_qs = src_quotes),
                         "verified in source",
                         "FABRICATED / UNVERIFIED")
)
print(quote_audit)



# 2.7 Self-consistency over 5 samples


llm_summarize_stub_temperature <- function(article_body, seed) {
  set.seed(seed)
  base <- llm_summarize_stub(article_body)
  altered_num <- sample(c("$310 billion","$285 billion","$330 billion","$310 billion","$295 billion"), 1)
  str_replace(base, "\\$310 billion", altered_num)
}

samples <- map_chr(1:5, ~ llm_summarize_stub_temperature(article$body, seed = .x))

jaccard <- function(a, b) {
  A <- unique(tok(a)); B <- unique(tok(b))
  if (length(union(A, B)) == 0) return(0)
  length(intersect(A, B)) / length(union(A, B))
}

pair_agreement <- combn(samples, 2, simplify = FALSE) %>%
  map_dbl(~ jaccard(.x[1], .x[2]))

print(tibble(
  n_samples      = length(samples),
  mean_agreement = mean(pair_agreement),
  hallu_score    = 1 - mean(pair_agreement)
))



# PART III — MODEL SELECTION FOR NEWS ANNOTATION


topics <- c("politics","economy","climate","legal","opinion")

simulate_llm <- function(true_labels, accuracy, alternatives) {
  n <- length(true_labels)
  ifelse(runif(n) < accuracy, true_labels,
         sample(alternatives, n, replace = TRUE))
}

set.seed(6323)
model_predictions <- gold %>%
  mutate(
    pred_frontier  = simulate_llm(topic, 0.92, topics),
    pred_workhorse = simulate_llm(topic, 0.85, topics),
    pred_small     = simulate_llm(topic, 0.72, topics)
  )

evaluate_tier <- function(df, pred_col, label = pred_col) {
  df2 <- df %>%
    mutate(truth = factor(topic,          levels = topics),
           pred  = factor(.data[[pred_col]], levels = topics))
  tibble(
    model    = label,
    accuracy = mean(df[[pred_col]] == df$topic),
    kappa    = irr::kappa2(cbind(df[[pred_col]], df$topic))$value,
    macro_f1 = f_meas(df2, truth = truth, estimate = pred, estimator = "macro")$.estimate
  )
}

tier_results <- bind_rows(
  evaluate_tier(model_predictions, "pred_frontier",  "frontier"),
  evaluate_tier(model_predictions, "pred_workhorse", "workhorse"),
  evaluate_tier(model_predictions, "pred_small",     "small")
)

pricing <- tibble(
  model     = c("frontier", "workhorse", "small"),
  price_in  = c(15.00, 3.00, 0.25),
  price_out = c(75.00, 15.00, 1.25)
)

scenario <- list(n_items = 500000, tokens_in = 60, tokens_out = 8)

cost_analysis <- tier_results %>%
  left_join(pricing, by = "model") %>%
  mutate(
    total_cost       = (scenario$n_items * scenario$tokens_in  / 1e6) * price_in +
                       (scenario$n_items * scenario$tokens_out / 1e6) * price_out,
    cost_per_correct = total_cost / (accuracy * scenario$n_items)
  )
print(cost_analysis)

p_frontier <- cost_analysis %>%
  ggplot(aes(x = total_cost, y = kappa, label = model, color = model)) +
  geom_point(size = 6) +
  geom_text(vjust = -1.2, size = 5) +
  scale_x_continuous(labels = scales::dollar) +
  labs(x = "Total cost for 500K headlines (USD)",
       y = expression("Cohen's"~kappa~"vs. gold labels"),
       title = "Quality-Cost Frontier for Headline Classification",
       subtitle = "Five-year political news corpus, 500,000 items") +
  theme(legend.position = "none")
print(p_frontier)
!
