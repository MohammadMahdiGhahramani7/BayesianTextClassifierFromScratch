class My_NB_Text_Classifier:

  def __init__(self, sens, stop_words):

    self.sens = {i.lower():sens[i] for i in sens}
    self.stw = stop_words
    self.prob_positive = sum(sens.values()) / len(sens)
    self.prob_negative = 1 - self.prob_positive
    self.positives, self.positive_words_p_hats = {}, {}
    self.negatives, self.negative_words_p_hats = {}, {}
    self.extra = list("!@#$%^&*()./-+~0123456789")

  def _remove_spaces_of_lists(self, words):

    while '' in words:

      words.pop(words.index(''))

    return words

  def _clean_and_remove_stop_words_(self, sen):

    #Cleaning
    for sym in self.extra:

      sen = sen.replace(sym, "")

    #Extract words
    words = sen.split(" ")

    #removing stop word
    for w in words:

      if w in self.stw:

        words[words.index(w)] = ""

    return words

  def _add_word_to_pn(self, words, label):

    if label:

      for w in words:

        if w in self.positives:

          self.positives[w] += 1

        else:

          self.positives[w] = 1

    else:

      for w in words:

        if w in self.negatives:

          self.negatives[w] += 1

        else:

          self.negatives[w] = 1

  def fit(self):

    for key in self.sens:

      sentence, label = key, self.sens[key]

      words = self._clean_and_remove_stop_words_(sentence)

      self._add_word_to_pn(words, label)

    self.positives.pop('')
    self.negatives.pop('')

  def predict(self, new_sen, verbose=True):

    positives_copy, negatives_copy = self.positives, self.negatives

    new_sen = new_sen.lower()

    ws = self._clean_and_remove_stop_words_(new_sen)
    
    ws = self._remove_spaces_of_lists(ws)

    self.being_positive, self.being_negative = 1, 1

    #Evaluate of being POSITIVE
    for w in ws:

      if w not in positives_copy:

        positives_copy[w] = 0

    #Pˆ(wk |vj) = (nk + 1) / (n + |Vocabulary|)
    positive_length = sum(positives_copy.values())
    positive_vocab_length = len(positives_copy.values())

    for wp in positives_copy:

      self.positive_words_p_hats[wp] = (positives_copy[wp] + 1) / (positive_length + positive_vocab_length)

    for w in ws:

      self.being_positive *= self.positive_words_p_hats[w]


    #Evaluate of being NEGATIVE
    for w in ws:

      if w not in negatives_copy:

        negatives_copy[w] = 0

    #Pˆ(wk |vj) = (nk + 1) / (n + |Vocabulary|)
    negative_length = sum(negatives_copy.values())
    negative_vocab_length = len(negatives_copy.values())

    for wp in negatives_copy:

      self.negative_words_p_hats[wp] = (negatives_copy[wp] + 1) / (negative_length + negative_vocab_length)

    for w in ws:

      self.being_negative *= self.negative_words_p_hats[w]

    self.prob_of_being_postive = self.being_positive / (self.being_positive + self.being_negative)
    self.prob_of_being_negative = 1 - self.prob_of_being_postive


    if self.being_positive > self.being_negative:

      if verbose:

        print(f"POSITIVE\nProbability(positive): {self.prob_of_being_postive:.3f}\nProbability(negative): {self.prob_of_being_negative:.3f}")

      return 1

    else:

      if verbose:

        print(f"NEGATIVE\nProbability(positive): {self.prob_of_being_postive:.3f}\nProbability(negative): {self.prob_of_being_negative:.3f}")

      return 0
