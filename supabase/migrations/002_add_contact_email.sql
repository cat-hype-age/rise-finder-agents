-- Add contact_email column to composite_scores
-- Stores maintainer email discovered from npm/PyPI/GitHub profiles
ALTER TABLE composite_scores ADD COLUMN IF NOT EXISTS contact_email TEXT DEFAULT '';
