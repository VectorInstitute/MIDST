---
layout: custom 
title: MIDST Challenge @ SaTML 2025
permalink: /sponsors/
---
# Sponsors 
<style>
p, ol, ul, li {
  color: #000000 !important
}


.sponsors-page .sponsors-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  padding: 20px;
}

.sponsors-page .sponsor {
  text-align: center;
  border: 1px solid #ddd;
  padding: 10px;
  background-color: #f9f9f9;
}

.sponsors-page .sponsor img {
  height: 150px;      /* Set a fixed height */
  object-fit: cover;  /* Scale the image to cover the container */
}

.sponsors-page .sponsor p {
  margin-top: 10px;
  font-size: 1.2rem;
}

</style>

<div class="sponsors-page">

  <div class="sponsors-grid">
    {% for sponsor in site.data.sponsors %}
    <div class="sponsor">
<img src="{{ '/assets/images/sponsors/' | relative_url }}{{ sponsor.image }}" alt="{{ sponsor.name }}">
      <p>{{ sponsor.name }}</p>
    </div>
    {% endfor %}
  </div>
</div>

[Back to Main Page]({{ '/' | relative_url }})
