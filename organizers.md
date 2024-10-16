---
layout: default
title: Event Organizers
permalink: /organizers/
---
<style>

.organizers-page .organizer-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  padding: 20px;
}

.organizers-page .organizer {
  text-align: center;
  border: 1px solid #ddd;
  padding: 10px;
  background-color: #f9f9f9;
}

.organizers-page .organizer img {
  width: 150px;       /* Set a fixed width */
  height: 150px;      /* Set a fixed height */
  object-fit: cover;  /* Scale the image to cover the container */
  border-radius: 50%;
}

.organizers-page .organizer p {
  margin-top: 10px;
  font-size: 1.2rem;
}

</style>
# Event Organizers

<div class="organizers-page">

  <div class="organizer-grid">
    {% for organizer in site.data.organizers %}
    <div class="organizer">
<img src="{{ '/assets/images/' | relative_url }}{{ organizer.image }}" alt="{{ organizer.name }}">
      <p>{{ organizer.name }}</p>
    </div>
    {% endfor %}
  </div>
</div>
