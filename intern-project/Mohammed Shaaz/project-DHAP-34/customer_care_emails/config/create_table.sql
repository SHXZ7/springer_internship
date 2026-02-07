CREATE TABLE IF NOT EXISTS public.customer_care_emails (
  subject TEXT NOT NULL,
  sender TEXT NOT NULL,
  receiver TEXT NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  message_body TEXT NOT NULL,
  thread_id TEXT NOT NULL,
  email_types TEXT,
  email_status TEXT,
  email_criticality TEXT,
  product_types TEXT,
  agent_effectivity TEXT,
  agent_efficiency TEXT,
  customer_satisfaction NUMERIC,
  PRIMARY KEY (thread_id)
);
